import json
from pathlib import Path

import numpy as np
import pytest

from delta_vision.actuator import DryRunActuator
from delta_vision.calibrate_templates import MatcherConfiguration, calibrate_templates
from delta_vision.evaluate_templates import evaluate_template_profile
from delta_vision.events import JsonlEventWriter
from delta_vision.frames import CapturedFrame, FrameRecorder, ReplayFrameSource
from delta_vision.navigation import NavigationStatus
from delta_vision.worker import (
    build_navigation_controller,
    load_worker_settings,
    run_control_loop,
)

FRAME_WIDTH = 100
FRAME_HEIGHT = 80
ROI = {"id": "scene", "x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}


def _base_image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        256,
        size=(FRAME_HEIGHT, FRAME_WIDTH, 3),
        dtype=np.uint8,
    )


def _blind_variant(image: np.ndarray, delta: int) -> np.ndarray:
    variant = np.array(image, copy=True)
    roi = variant[8:72, 10:90].astype(np.int16)
    variant[8:72, 10:90] = np.clip(roi + delta, 0, 255).astype(np.uint8)
    variant[:8, :] = delta
    return variant


def _record_dataset(
    root: Path,
    *,
    run_id: str,
    split: str,
    images: list[np.ndarray],
) -> None:
    recorder = FrameRecorder(root)
    for sequence, image in enumerate(images):
        owned = np.array(image, copy=True)
        owned.setflags(write=False)
        recorder.record(
            CapturedFrame(
                sequence=sequence,
                captured_at_ns=1_000 + sequence,
                image=owned,
                source="offline-e2e",
            ),
            metadata={
                "run_id": run_id,
                "dataset_kind": "manual-game-route",
                "dataset_split": split,
            },
        )
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset_split": split,
                "window_title": "三角洲行动",
                "backend": "offline-e2e",
                "frame_count": len(images),
                "resolution": [FRAME_WIDTH, FRAME_HEIGHT],
            }
        ),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_worker_config(path: Path, profile_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "armed_ready": False,
                "target_window_title": "三角洲行动",
                "capture_backend": "dxcam",
                "emergency_virtual_key": 123,
                "max_key_hold_ms": 250,
                "loop_interval_ms": 100,
                "max_duration_seconds": 15,
                "perception": {
                    "mode": "template",
                    "template_profile": profile_path.relative_to(path.parent).as_posix(),
                },
                "goal_node_id": "goal",
                "nodes": {
                    "start": {
                        "x": 0,
                        "y": 0,
                        "edges": [{"target": "turn", "cost": 1}],
                    },
                    "turn": {
                        "x": 100,
                        "y": 0,
                        "edges": [{"target": "goal", "cost": 1}],
                    },
                    "goal": {"x": 200, "y": 0, "edges": []},
                },
                "edge_actions": [
                    {
                        "source": "start",
                        "target": "turn",
                        "key": "w",
                        "mouse_dx": 40,
                        "mouse_dy": 0,
                    },
                    {
                        "source": "turn",
                        "target": "goal",
                        "key": "w",
                        "mouse_dx": 80,
                        "mouse_dy": 0,
                    },
                ],
                "navigation": {
                    "pulse_ms": 80,
                    "min_progress_px": 5,
                    "stuck_after_ms": 600,
                    "localization_timeout_ms": 800,
                    "max_recovery_attempts": 0,
                    "recovery_keys": [],
                    "arrival_confirmations": 2,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_calibration_evaluation_and_worker_replay_form_one_pipeline(tmp_path) -> None:
    base_images = [_base_image(seed) for seed in (11, 22, 33)]
    calibration_root = tmp_path / "calibration"
    _record_dataset(
        calibration_root,
        run_id="route-e2e-cal",
        split="calibration",
        images=base_images,
    )
    calibration_labels = tmp_path / "calibration-labels.jsonl"
    _write_jsonl(
        calibration_labels,
        [
            {
                "run_id": "route-e2e-cal",
                "sequence": sequence,
                "split": "calibration",
                "locatable": True,
                "template_id": waypoint_id,
                "roi": ROI,
                "route_position": [sequence * 100, 0],
                "waypoint_id": waypoint_id,
            }
            for sequence, waypoint_id in enumerate(("start", "turn", "goal"))
        ],
    )
    profile_root = tmp_path / "profile"
    calibration = calibrate_templates(
        dataset_directory=calibration_root,
        labels_path=calibration_labels,
        output_directory=profile_root,
        matcher=MatcherConfiguration.default(),
    )

    blind_images = [
        _blind_variant(base_images[0], 1),
        _blind_variant(base_images[1], 2),
        _blind_variant(base_images[2], 3),
        _blind_variant(base_images[2], 4),
        np.zeros_like(base_images[0]),
    ]
    blind_root = tmp_path / "blind"
    _record_dataset(
        blind_root,
        run_id="route-e2e-blind",
        split="blind",
        images=blind_images,
    )
    blind_labels = tmp_path / "blind-labels.jsonl"
    positions = ((0, 0), (100, 0), (200, 0), (200, 0), None)
    waypoint_ids = ("start", "turn", "goal", "goal", None)
    _write_jsonl(
        blind_labels,
        [
            {
                "run_id": "route-e2e-blind",
                "sequence": sequence,
                "split": "blind",
                "locatable": position is not None,
                "route_position": list(position) if position is not None else None,
                "expected_waypoint_id": waypoint_id,
            }
            for sequence, (position, waypoint_id) in enumerate(
                zip(positions, waypoint_ids, strict=True)
            )
        ],
    )
    evaluation = evaluate_template_profile(
        profile_path=calibration.manifest_path,
        dataset_directory=blind_root,
        labels_path=blind_labels,
        output_directory=tmp_path / "evaluation",
        split="blind",
        distance_tolerance=25,
    )
    metrics = json.loads(evaluation.metrics_path.read_text(encoding="utf-8"))

    config_path = tmp_path / "worker.json"
    _write_worker_config(config_path, calibration.manifest_path)
    settings = load_worker_settings(config_path)
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)
    controller = build_navigation_controller(settings, actuator=actuator)
    current_time_ns = 0

    def clock_ns() -> int:
        return current_time_ns

    def sleep_fn(seconds: float) -> None:
        nonlocal current_time_ns
        current_time_ns += int(seconds * 1_000_000_000)

    source = ReplayFrameSource(blind_root)
    result = run_control_loop(
        source=source,
        controller=controller,
        actuator=actuator,
        recorder=FrameRecorder(tmp_path / "worker-replay"),
        event_writer=JsonlEventWriter(tmp_path / "worker-events.jsonl"),
        loop_interval_ms=settings.loop_interval_ms,
        max_duration_seconds=settings.max_duration_seconds,
        clock_ns=clock_ns,
        sleep_fn=sleep_fn,
    )

    assert metrics["waypoint_top1_accuracy"] == 1
    assert metrics["pose_emission_f1"] == 1
    assert metrics["false_lock_rate"] == 0
    assert metrics["unlocatable_count"] == 1
    assert result.status is NavigationStatus.ARRIVED
    assert [(event.kind, event.key, event.dx, event.dy) for event in actuator.events] == [
        ("mouse_move", None, 40, 0),
        ("key_down", "w", None, None),
        ("key_up", "w", None, None),
        ("mouse_move", None, 80, 0),
        ("key_down", "w", None, None),
        ("key_up", "w", None, None),
    ]
    assert actuator.pressed_keys == frozenset()
    with pytest.raises(RuntimeError, match="已经关闭"):
        source.grab()
    replay_input_records = [
        json.loads(line)
        for line in (tmp_path / "worker-replay" / "input-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["payload"]["kind"] for record in replay_input_records] == [
        event.kind for event in actuator.events
    ]
