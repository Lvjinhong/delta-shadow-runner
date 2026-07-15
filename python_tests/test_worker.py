import json
from pathlib import Path

import numpy as np
import pytest

from delta_vision.actuator import DryRunActuator
from delta_vision.events import JsonlEventWriter
from delta_vision.frames import CapturedFrame, FrameRecorder, ReplayFrameSource
from delta_vision.navigation import NavigationStatus
from delta_vision.worker import (
    build_navigation_controller,
    load_worker_settings,
    run_control_loop,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "controlled-window.json"


def _frame(sequence: int, x: int, y: int) -> CapturedFrame:
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[y - 10 : y + 10, x - 10 : x + 10] = (0, 255, 0)
    image.setflags(write=False)
    return CapturedFrame(sequence, 1_000 + sequence, image, "fixture")


class FakeFrameSource:
    def __init__(self, frames) -> None:
        self._frames = iter(frames)
        self.closed = False

    def grab(self):
        return next(self._frames)

    def close(self) -> None:
        self.closed = True


def test_load_controlled_window_settings_from_json() -> None:
    settings = load_worker_settings(CONFIG_PATH)

    assert settings.target_window_title == "Delta Vision Test Target"
    assert settings.capture_backend == "dxcam"
    assert settings.goal_node_id == "goal"
    assert settings.graph["start"].edges[0].target_node_id == "turn"
    assert settings.policy.edge_actions[("turn", "goal")] == "d"
    assert settings.marker_bgr == (0, 255, 0)
    assert settings.max_duration_seconds == 15


def test_load_worker_settings_rejects_unknown_schema(tmp_path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["schema_version"] = 2
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_worker_settings(path)


def test_load_worker_settings_accepts_zero_screen_coordinates(tmp_path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["nodes"]["start"]["x"] = 0
    config["nodes"]["start"]["y"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    settings = load_worker_settings(path)

    assert settings.graph["start"].x == 0
    assert settings.graph["start"].y == 0


@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (lambda config: config.update(max_duration_seconds=float("inf")), "正有限数"),
        (
            lambda config: config["marker"].update(confidence_threshold=1.1),
            "confidence_threshold",
        ),
        (
            lambda config: config["navigation"].update(max_recovery_attempts="2"),
            "max_recovery_attempts",
        ),
    ],
)
def test_load_worker_settings_rejects_unsafe_numeric_values(
    tmp_path, mutate, error_match: str
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_worker_settings(path)


def test_control_loop_reaches_goal_from_screenshot_frames_and_records_replay(
    tmp_path,
) -> None:
    settings = load_worker_settings(CONFIG_PATH)
    actuator = DryRunActuator(
        allowed_keys={"w", "a", "s", "d"},
        max_key_hold_ms=250,
    )
    controller = build_navigation_controller(settings, actuator=actuator)
    source = FakeFrameSource(
        [
            _frame(0, 80, 520),
            _frame(1, 80, 80),
            _frame(2, 700, 80),
            _frame(3, 700, 80),
        ]
    )
    clock = iter([0, 0, 100_000_000, 200_000_000, 250_000_000, 300_000_000])

    result = run_control_loop(
        source=source,
        controller=controller,
        actuator=actuator,
        recorder=FrameRecorder(tmp_path / "replay"),
        event_writer=JsonlEventWriter(tmp_path / "events.jsonl"),
        clock_ns=lambda: next(clock),
        sleep_fn=lambda _: None,
        loop_interval_ms=20,
        max_duration_seconds=15,
    )

    assert result.status is NavigationStatus.ARRIVED
    assert result.frame_count == 4
    assert source.closed is True
    assert actuator.pressed_keys == frozenset()
    assert len(list(ReplayFrameSource(tmp_path / "replay"))) == 4
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["payload"]["status"] for event in events] == [
        "navigating",
        "navigating",
        "navigating",
        "arrived",
    ]
    assert events[-1]["payload"]["pressed_keys"] == []


def test_control_loop_closes_source_and_releases_keys_on_capture_error(tmp_path) -> None:
    settings = load_worker_settings(CONFIG_PATH)
    actuator = DryRunActuator(
        allowed_keys={"w", "a", "s", "d"},
        max_key_hold_ms=250,
    )
    controller = build_navigation_controller(settings, actuator=actuator)

    class FailingSource(FakeFrameSource):
        def grab(self):
            if actuator.pressed_keys:
                raise RuntimeError("capture failed")
            return _frame(0, 80, 520)

    source = FailingSource([])
    clock = iter([0, 0, 10_000_000, 20_000_000])

    with pytest.raises(RuntimeError, match="capture failed"):
        run_control_loop(
            source=source,
            controller=controller,
            actuator=actuator,
            recorder=FrameRecorder(tmp_path / "replay"),
            event_writer=JsonlEventWriter(tmp_path / "events.jsonl"),
            clock_ns=lambda: next(clock),
            sleep_fn=lambda _: None,
            loop_interval_ms=20,
            max_duration_seconds=15,
        )

    assert source.closed is True
    assert actuator.pressed_keys == frozenset()
