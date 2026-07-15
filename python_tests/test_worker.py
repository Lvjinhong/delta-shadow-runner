import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from delta_vision.actuator import DryRunActuator
from delta_vision.config import CaptureRegion
from delta_vision.controlled_target import GOAL_RADIUS
from delta_vision.events import JsonlEventWriter
from delta_vision.frames import (
    CapturedFrame,
    DatasetContentDigest,
    FrameRecorder,
    ReplayFrameSource,
)
from delta_vision.navigation import NavigationStatus
from delta_vision.worker import (
    build_navigation_controller,
    build_windows_runtime,
    load_worker_settings,
    run_control_loop,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "controlled-window.json"


def _frame(sequence: int, x: int, y: int) -> CapturedFrame:
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[y - 10 : y + 10, x - 10 : x + 10] = (0, 255, 0)
    image.setflags(write=False)
    return CapturedFrame(sequence, 1_000 + sequence, image, "fixture")


def _write_template_image(path: Path, seed: int) -> tuple[np.ndarray, str]:
    image = np.random.default_rng(seed).integers(
        0, 256, size=(12, 16, 3), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return image, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_template_worker_fixture(
    tmp_path: Path,
) -> tuple[Path, np.ndarray, np.ndarray]:
    config_root = tmp_path / "configs"
    profile_root = config_root / "route-01"
    start_image, start_hash = _write_template_image(
        profile_root / "start.png", 11
    )
    goal_image, goal_hash = _write_template_image(profile_root / "goal.png", 12)
    dataset_digest = DatasetContentDigest()
    dataset_digest.update_hash(10, "1" * 64)
    dataset_digest.update_hash(20, "2" * 64)
    profile = {
        "schema_version": 2,
        "capture_profile": {"width": 180, "height": 120},
        "matcher": {
            "scales": [1.0],
            "score_threshold": 0.8,
            "minimum_spatial_margin": 0.05,
            "minimum_template_margin": 0.05,
            "nms_radius_px": 18,
        },
        "rois": {
            "scene": {"left": 0, "top": 0, "width": 180, "height": 120}
        },
        "source_datasets": [
            {
                "run_id": "calibration-run-01",
                "frame_sha256s": ["1" * 64, "2" * 64],
                "frame_hashes": [
                    {"sequence": 10, "sha256": "1" * 64},
                    {"sequence": 20, "sha256": "2" * 64},
                ],
                "perception_sha256s": ["a" * 64, "b" * 64],
                "dataset_content_sha256": dataset_digest.hexdigest(),
                "run_json_sha256": "c" * 64,
                "frame_manifest_sha256": "d" * 64,
            }
        ],
        "templates": [
            {
                "id": "start-template",
                "image": "start.png",
                "sha256": start_hash,
                "roi_id": "scene",
                "route_position": [0, 0],
                "waypoint_id": "start",
                "source_run_id": "calibration-run-01",
                "source_sequence": 10,
                "source_frame_sha256": "1" * 64,
            },
            {
                "id": "goal-template",
                "image": "goal.png",
                "sha256": goal_hash,
                "roi_id": "scene",
                "route_position": [100, 0],
                "waypoint_id": "goal",
                "source_run_id": "calibration-run-01",
                "source_sequence": 20,
                "source_frame_sha256": "2" * 64,
            },
        ],
    }
    profile_path = profile_root / "templates.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    worker = {
        "schema_version": 2,
        "target_window_title": "三角洲行动",
        "capture_backend": "dxcam",
        "emergency_virtual_key": 123,
        "max_key_hold_ms": 250,
        "loop_interval_ms": 20,
        "max_duration_seconds": 15,
        "perception": {
            "mode": "template",
            "template_profile": "route-01/templates.json",
        },
        "goal_node_id": "goal",
        "nodes": {
            "start": {
                "x": 0,
                "y": 0,
                "edges": [{"target": "goal", "cost": 1}],
            },
            "goal": {"x": 100, "y": 0, "edges": []},
        },
        "edge_actions": [
            {
                "source": "start",
                "target": "goal",
                "key": "w",
                "mouse_dx": 320,
                "mouse_dy": -12,
            }
        ],
        "navigation": {
            "pulse_ms": 100,
            "min_progress_px": 4,
            "stuck_after_ms": 600,
            "localization_timeout_ms": 800,
            "max_recovery_attempts": 0,
            "recovery_keys": [],
            "arrival_confirmations": 2,
        },
    }
    config_path = config_root / "game-route.json"
    config_path.write_text(json.dumps(worker), encoding="utf-8")
    return config_path, start_image, goal_image


def _template_frame(sequence: int, template: np.ndarray) -> CapturedFrame:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    image[40:52, 60:76] = template
    image.setflags(write=False)
    return CapturedFrame(sequence, 1_000 + sequence, image, "game-fixture")


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
    legacy_action = settings.policy.edge_actions[("turn", "goal")]
    assert (legacy_action.key, legacy_action.mouse_dx, legacy_action.mouse_dy) == ("d", 0, 0)
    assert settings.perception.bgr == (0, 255, 0)
    assert settings.max_duration_seconds == 15
    assert settings.max_key_hold_ms == 250
    assert settings.perception.localization_radius <= GOAL_RADIUS


def test_load_worker_settings_rejects_unknown_schema(tmp_path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["schema_version"] = 3
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_worker_settings(path)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_load_worker_settings_requires_exact_integer_schema(
    tmp_path, schema_version: object
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["schema_version"] = schema_version
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_worker_settings(path)


def test_load_worker_settings_v2_resolves_template_profile_relative_to_config(
    tmp_path,
) -> None:
    config_path, _, _ = _write_template_worker_fixture(tmp_path)

    settings = load_worker_settings(config_path)

    assert settings.target_window_title == "三角洲行动"
    assert settings.perception.frame_size == (180, 120)
    assert settings.perception.source_run_ids == frozenset({"calibration-run-01"})
    action = settings.policy.edge_actions[("start", "goal")]
    assert (action.key, action.mouse_dx, action.mouse_dy) == ("w", 320, -12)


@pytest.mark.parametrize(
    "reference",
    ["", "/tmp/templates.json", "C:\\templates.json", "C:templates.json", "\\\\host\\x.json"],
)
def test_load_worker_settings_v2_rejects_unsafe_template_profile_reference(
    tmp_path, reference: str
) -> None:
    config_path, _, _ = _write_template_worker_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["perception"]["template_profile"] = reference
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="template_profile"):
        load_worker_settings(config_path)


def test_load_worker_settings_v2_rejects_non_template_perception_mode(tmp_path) -> None:
    config_path, _, _ = _write_template_worker_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["perception"]["mode"] = "color_anchor"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"perception\.mode"):
        load_worker_settings(config_path)


def test_template_profile_drives_worker_controller_to_goal(tmp_path) -> None:
    config_path, start_image, goal_image = _write_template_worker_fixture(tmp_path)
    settings = load_worker_settings(config_path)
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)
    controller = build_navigation_controller(settings, actuator=actuator)

    first = controller.on_frame(_template_frame(0, start_image), now_ns=0)
    second = controller.on_frame(_template_frame(1, goal_image), now_ns=100_000_000)
    third = controller.on_frame(_template_frame(2, goal_image), now_ns=200_000_000)

    assert first.status is NavigationStatus.NAVIGATING
    assert second.status is NavigationStatus.NAVIGATING
    assert third.status is NavigationStatus.ARRIVED
    assert (actuator.events[0].kind, actuator.events[0].dx, actuator.events[0].dy) == (
        "mouse_move",
        320,
        -12,
    )
    assert actuator.pressed_keys == frozenset()


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
        (
            lambda config: config["marker"].update(tolerance=256),
            "marker.tolerance",
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


@pytest.mark.parametrize("value", [True, 1.5, 4097, -4097])
def test_load_worker_settings_rejects_unsafe_relative_mouse_delta(
    tmp_path, value
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["edge_actions"][0]["mouse_dx"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="鼠标"):
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
    frame_events = [event for event in events if event["event_type"] == "frame"]
    assert [event["payload"]["status"] for event in frame_events] == [
        "navigating",
        "navigating",
        "navigating",
        "arrived",
    ]
    assert frame_events[-1]["payload"]["pressed_keys"] == []


def test_control_loop_records_mouse_and_key_events_in_replay_and_event_log(
    tmp_path,
) -> None:
    config_path, start_image, goal_image = _write_template_worker_fixture(tmp_path)
    settings = load_worker_settings(config_path)
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)
    controller = build_navigation_controller(settings, actuator=actuator)
    source = FakeFrameSource(
        [
            _template_frame(0, start_image),
            None,
            _template_frame(1, goal_image),
            _template_frame(2, goal_image),
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

    replayed = list(ReplayFrameSource(tmp_path / "replay"))
    assert result.status is NavigationStatus.ARRIVED
    assert [event["kind"] for event in replayed[0].metadata["input_events"]] == [
        "mouse_move",
        "key_down",
    ]
    assert replayed[0].metadata["input_events"][0]["dx"] == 320
    assert [event["kind"] for event in replayed[1].metadata["input_events"]] == [
        "key_up"
    ]
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["payload"]["kind"] for event in events if event["event_type"] == "input"] == [
        "mouse_move",
        "key_down",
        "key_up",
    ]


def test_control_loop_checks_overdue_keys_before_every_capture(tmp_path) -> None:
    settings = load_worker_settings(CONFIG_PATH)

    class TrackingActuator(DryRunActuator):
        def __init__(self) -> None:
            super().__init__(
                allowed_keys={"w", "a", "s", "d"},
                max_key_hold_ms=250,
            )
            self.expiry_checks: list[int] = []

        def expire_overdue(self, *, now_ns: int) -> tuple[str, ...]:
            self.expiry_checks.append(now_ns)
            return super().expire_overdue(now_ns=now_ns)

    actuator = TrackingActuator()
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
    assert actuator.expiry_checks == [0, 100_000_000, 200_000_000, 250_000_000]


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


class FakeGateway:
    def __init__(self) -> None:
        self.sent = []

    def foreground_window_handle(self) -> int:
        return 123

    def foreground_title(self) -> str:
        return "Delta Vision Test Target"

    def is_key_pressed(self, virtual_key: int) -> bool:
        assert virtual_key == 123
        return False

    def send_key(self, scan_code: int, *, key_up: bool) -> int:
        self.sent.append((scan_code, key_up))
        return 1

    def send_mouse_relative(self, dx: int, dy: int) -> int:
        return 1


def test_build_windows_runtime_defaults_to_dry_run(tmp_path) -> None:
    settings = load_worker_settings(CONFIG_PATH)
    source = FakeFrameSource([])
    resolver_calls = []

    def resolve_region(_: str) -> CaptureRegion:
        resolver_calls.append("dpi-aware-region")
        return CaptureRegion(0, 0, 800, 600)

    def resolve_handle(_: str) -> int:
        resolver_calls.append("window-handle")
        return 123

    runtime = build_windows_runtime(
        settings,
        artifacts=tmp_path,
        armed=False,
        window_handle_resolver=resolve_handle,
        region_resolver=resolve_region,
        dxcam_factory=lambda region: source,
        gateway_factory=lambda: (_ for _ in ()).throw(
            AssertionError("dry-run 不应加载输入 gateway")
        ),
    )

    assert isinstance(runtime.actuator, DryRunActuator)
    assert runtime.source is source
    assert runtime.target_window_handle == 123
    assert resolver_calls == ["dpi-aware-region", "window-handle"]


def test_build_windows_runtime_rejects_template_profile_resolution_mismatch(
    tmp_path,
) -> None:
    config_path, _, _ = _write_template_worker_fixture(tmp_path)
    settings = load_worker_settings(config_path)

    with pytest.raises(ValueError, match=r"模板 Profile.*180x120.*800x600"):
        build_windows_runtime(
            settings,
            artifacts=tmp_path / "artifacts",
            armed=False,
            window_handle_resolver=lambda _: (_ for _ in ()).throw(
                AssertionError("分辨率不匹配时不应解析 HWND")
            ),
            region_resolver=lambda _: CaptureRegion(0, 0, 800, 600),
            dxcam_factory=lambda _: (_ for _ in ()).throw(
                AssertionError("分辨率不匹配时不应创建截图 backend")
            ),
        )


def test_build_windows_runtime_rejects_unsupported_route_action_key(tmp_path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["edge_actions"][0]["key"] = "x"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    settings = load_worker_settings(config_path)

    with pytest.raises(ValueError, match=r"不支持的按键.*x"):
        build_windows_runtime(
            settings,
            artifacts=tmp_path / "artifacts",
            armed=False,
            region_resolver=lambda _: (_ for _ in ()).throw(
                AssertionError("非法按键应在截图初始化前被拒绝")
            ),
        )


def test_build_windows_runtime_armed_uses_bound_win32_safety_gate(tmp_path) -> None:
    settings = load_worker_settings(CONFIG_PATH)
    source = FakeFrameSource([])
    gateway = FakeGateway()

    runtime = build_windows_runtime(
        settings,
        artifacts=tmp_path,
        armed=True,
        window_handle_resolver=lambda _: 123,
        region_resolver=lambda _: CaptureRegion(0, 0, 800, 600),
        dxcam_factory=lambda region: source,
        gateway_factory=lambda: gateway,
    )
    runtime.actuator.key_down("w", now_ns=1)
    runtime.actuator.key_up("w", now_ns=2)

    assert gateway.sent == [(0x11, False), (0x11, True)]
    assert runtime.actuator.pressed_keys == frozenset()
