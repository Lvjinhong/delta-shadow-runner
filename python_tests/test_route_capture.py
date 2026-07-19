import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from delta_vision.config import CaptureRegion
from delta_vision.events import JsonlEventWriter
from delta_vision.frames import CapturedFrame, FrameRecorder, ReplayFrameSource
from delta_vision.menu_automation import (
    MenuScene,
    SceneDecisionReason,
    SceneObservation,
)
from delta_vision.route_capture import (
    GuardedPulseExecutor,
    RouteCaptureRuntime,
    RouteCaptureSettings,
    RouteCaptureStatus,
    RouteCaptureStep,
    build_windows_route_capture_runtime,
    canonical_route_capture_plan,
    load_route_capture_settings,
    main,
    route_capture_plan_sha256,
    run_route_capture_loop,
)
from delta_vision.route_capture_windows import ExactWindowGuard
from delta_vision.safe_input import EmergencyStopError, InputEvent


@dataclass(frozen=True)
class _Profile:
    observer: object
    frame_size: tuple[int, int] = (1920, 1080)
    profile_id: str = "menu-zero-cost-1080p-v1"
    profile_sha256: str = "a" * 64
    confirmation_frames: int = 3
    maximum_confirmation_span_ms: int = 500
    maximum_frame_age_ms: int = 500


def _write_config(
    tmp_path: Path,
    *,
    armed_ready: bool = False,
    steps: list[dict[str, object]] | None = None,
) -> Path:
    config_path = tmp_path / "configs" / "game-route.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "target_window_title": "三角洲行动  ",
                "capture_backend": "mss",
                "emergency_virtual_key": 123,
                "max_key_hold_ms": 250,
                "menu": {"profile": "../profiles/menu/menu.json"},
                "route_capture": {
                    "armed_ready": armed_ready,
                    "dataset_split": "calibration",
                    "guard_interval_ms": 20,
                    "max_duration_seconds": 30,
                    "steps": (
                        steps
                        if steps is not None
                        else [
                        {
                            "step_id": "spawn-forward-01",
                            "keys": ["w"],
                            "pulse_ms": 80,
                            "mouse_dx": 0,
                            "mouse_dy": 0,
                            "settle_ms": 200,
                        }
                        ]
                    ),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _settings(
    *,
    armed_ready: bool = True,
    steps: tuple[RouteCaptureStep, ...] | None = None,
) -> RouteCaptureSettings:
    observer = _Observer()
    return RouteCaptureSettings(
        target_window_title="三角洲行动  ",
        capture_backend="mss",
        emergency_virtual_key=123,
        max_key_hold_ms=250,
        menu_profile=_Profile(observer=observer),
        armed_ready=armed_ready,
        dataset_split="calibration",
        guard_interval_ms=20,
        max_duration_seconds=30,
        steps=steps
        or (
            RouteCaptureStep(
                step_id="spawn-forward-01",
                keys=("w",),
                pulse_ms=40,
                mouse_dx=5,
                mouse_dy=-2,
                settle_ms=20,
            ),
        ),
    )


class _Observer:
    def __init__(self, *, unknown_sequences: set[int] | None = None) -> None:
        self.unknown_sequences = unknown_sequences or set()

    def observe(self, frame: CapturedFrame) -> SceneObservation:
        accepted = frame.sequence not in self.unknown_sequences
        return SceneObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            scene=MenuScene.IN_MATCH if accepted else MenuScene.UNKNOWN,
            candidate_scene=MenuScene.IN_MATCH,
            confidence=0.99,
            runner_up_confidence=0.1,
            accepted=accepted,
            reason=(
                SceneDecisionReason.ACCEPTED
                if accepted
                else SceneDecisionReason.BELOW_THRESHOLD
            ),
            action_accepted=False,
            action_point=None,
            page_point=(100.0, 100.0) if accepted else None,
            page_template_id="in-match" if accepted else "in-match-candidate",
        )


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += round(seconds * 1_000_000_000)


class _Source:
    def __init__(self, frames: list[CapturedFrame], clock: _Clock) -> None:
        self.frames = iter(frames)
        self.clock = clock
        self.closed = False

    def grab(self):
        frame = next(self.frames, None)
        if frame is not None:
            self.clock.now_ns = max(self.clock.now_ns, frame.captured_at_ns)
        return frame

    def close(self) -> None:
        self.closed = True


class _Guard:
    def __init__(self) -> None:
        self.check_count = 0
        self.error: BaseException | None = None

    def check(self) -> None:
        self.check_count += 1
        if self.error is not None:
            raise self.error


class _Actuator:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self._events: list[InputEvent] = []
        self._pressed: set[str] = set()
        self.release_all_count = 0

    @property
    def events(self):
        return tuple(self._events)

    @property
    def pressed_keys(self):
        return frozenset(self._pressed)

    def move_mouse_relative(self, dx: int, dy: int, *, now_ns: int) -> None:
        self._events.append(InputEvent("mouse_move", now_ns, dx=dx, dy=dy))

    def key_down(self, key: str, *, now_ns: int) -> None:
        self._pressed.add(key)
        self._events.append(InputEvent("key_down", now_ns, key=key))

    def key_up(self, key: str, *, now_ns: int, reason: str | None = None) -> None:
        if key not in self._pressed:
            return
        self._pressed.remove(key)
        self._events.append(InputEvent("key_up", now_ns, key=key, reason=reason))

    def release_all(self, *, now_ns: int, reason: str) -> None:
        self.release_all_count += 1
        for key in tuple(self._pressed):
            self.key_up(key, now_ns=now_ns, reason=reason)


def _frame(sequence: int, captured_at_ms: int) -> CapturedFrame:
    image = np.full((1080, 1920, 3), sequence, dtype=np.uint8)
    image.setflags(write=False)
    return CapturedFrame(
        sequence=sequence,
        captured_at_ns=captured_at_ms * 1_000_000,
        image=image,
        source="fixture",
    )


def _runtime(
    tmp_path: Path,
    *,
    frames: list[CapturedFrame],
    clock: _Clock,
    actuator: _Actuator | None,
    observer: _Observer | None = None,
    guard: _Guard | None = None,
) -> RouteCaptureRuntime:
    root = tmp_path / "run"
    root.mkdir()
    return RouteCaptureRuntime(
        source=_Source(frames, clock),
        observer=observer or _Observer(),
        dataset_recorder=FrameRecorder(root / "dataset"),
        hud_recorder=FrameRecorder(root / "hud"),
        event_writer=JsonlEventWriter(root / "audit.jsonl", run_id="route-01"),
        actuator=actuator,
        guard=guard or _Guard(),
        target_window_handle=7,
        capture_region=CaptureRegion(320, 156, 1920, 1080),
        artifact_root=root,
    )


def test_load_settings_resolves_menu_profile_and_normalizes_plan(tmp_path) -> None:
    path = _write_config(tmp_path)
    loaded_paths: list[Path] = []

    settings = load_route_capture_settings(
        path,
        menu_profile_loader=lambda profile_path: (
            loaded_paths.append(Path(profile_path)) or _Profile(observer=_Observer())
        ),
    )

    assert loaded_paths == [(tmp_path / "profiles/menu/menu.json").resolve()]
    assert settings.steps[0].keys == ("w",)
    plan = canonical_route_capture_plan(settings)
    assert plan["frame_size"] == [1920, 1080]
    assert plan["steps"][0]["step_id"] == "spawn-forward-01"
    assert route_capture_plan_sha256(settings) == route_capture_plan_sha256(settings)


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [
            {
                "step_id": "dup",
                "keys": ["w"],
                "pulse_ms": 80,
                "settle_ms": 0,
            },
            {
                "step_id": "dup",
                "keys": ["d"],
                "pulse_ms": 80,
                "settle_ms": 0,
            },
        ],
        [{"step_id": "bad", "keys": ["enter"], "pulse_ms": 80, "settle_ms": 0}],
        [{"step_id": "bad", "keys": ["w"], "pulse_ms": 251, "settle_ms": 0}],
        [{"step_id": "bad", "keys": ["w"], "pulse_ms": 151, "settle_ms": 0}],
        [{"step_id": "bad", "keys": ["w"], "pulse_ms": True, "settle_ms": 0}],
        [
            {
                "step_id": "bad",
                "keys": ["w"],
                "pulse_ms": 80,
                "mouse_dx": 5000,
                "settle_ms": 0,
            }
        ],
    ],
)
def test_load_settings_rejects_unsafe_action_plans(tmp_path, steps) -> None:
    path = _write_config(tmp_path, steps=steps)

    with pytest.raises(ValueError):
        load_route_capture_settings(
            path,
            menu_profile_loader=lambda _path: _Profile(observer=_Observer()),
        )


def test_load_settings_rejects_duplicate_json_before_profile_load(tmp_path) -> None:
    path = tmp_path / "configs/game-route.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema_version":2,"schema_version":2}',
        encoding="utf-8",
    )
    calls: list[str] = []

    with pytest.raises(ValueError, match="重复字段"):
        load_route_capture_settings(
            path,
            menu_profile_loader=lambda _path: calls.append("profile"),
        )

    assert calls == []


@pytest.mark.parametrize("reference", ["/tmp/menu.json", "C:\\menu.json", "../../menu.json"])
def test_load_settings_rejects_menu_profile_outside_project(tmp_path, reference) -> None:
    path = _write_config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["menu"]["profile"] = reference
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="项目内相对路径"):
        load_route_capture_settings(
            path,
            menu_profile_loader=lambda _path: _Profile(observer=_Observer()),
        )


def test_settings_rejects_non_1080_profile() -> None:
    with pytest.raises(ValueError, match="1920x1080"):
        RouteCaptureSettings(
            target_window_title="三角洲行动  ",
            capture_backend="mss",
            emergency_virtual_key=123,
            max_key_hold_ms=250,
            menu_profile=_Profile(observer=_Observer(), frame_size=(1904, 1041)),
            armed_ready=False,
            dataset_split="calibration",
            guard_interval_ms=20,
            max_duration_seconds=30,
            steps=(RouteCaptureStep("step", ("w",), 80, 0, 0, 0),),
        )


def test_validate_only_does_not_open_window_create_artifacts_or_gateway(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _write_config(tmp_path)
    settings = _settings(armed_ready=False)
    calls: list[str] = []
    monkeypatch.setattr(
        "delta_vision.route_capture.load_route_capture_settings",
        lambda _path: calls.append("load") or settings,
    )
    monkeypatch.setattr(
        "delta_vision.route_capture.run_windows_route_capture",
        lambda *args, **kwargs: calls.append("runtime"),
    )

    artifacts = tmp_path / "must-not-exist"
    exit_code = main(
        ["--config", str(path), "--artifacts", str(artifacts), "--validate-only"]
    )

    assert exit_code == 0
    assert calls == ["load"]
    assert artifacts.exists() is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid"
    assert payload["plan_sha256"] == route_capture_plan_sha256(settings)


def test_cli_armed_requires_double_confirmation_before_config_load(
    tmp_path, monkeypatch, capsys
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "delta_vision.route_capture.load_route_capture_settings",
        lambda _path: calls.append("load"),
    )

    exit_code = main(
        [
            "--config",
            str(tmp_path / "missing.json"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--armed",
        ]
    )

    assert exit_code == 1
    assert calls == []
    assert "confirm-armed" in capsys.readouterr().err


def test_armed_ready_false_fails_before_artifact_window_or_gateway(tmp_path) -> None:
    calls: list[str] = []

    with pytest.raises(ValueError, match="armed_ready"):
        build_windows_route_capture_runtime(
            _settings(armed_ready=False),
            artifacts=tmp_path / "must-not-exist",
            armed=True,
            run_id="route-01",
            window_handle_resolver=lambda _title: calls.append("window") or 7,
            region_resolver=lambda _handle: CaptureRegion(0, 0, 1920, 1080),
            mss_factory=lambda _region: calls.append("source"),
            gateway_factory=lambda: calls.append("gateway"),
        )

    assert calls == []
    assert (tmp_path / "must-not-exist").exists() is False


def test_runtime_binds_exact_handle_and_1080p_without_gateway_in_dry_run(tmp_path) -> None:
    calls: list[object] = []
    source = _Source([], _Clock())

    runtime = build_windows_route_capture_runtime(
        _settings(),
        artifacts=tmp_path / "run",
        armed=False,
        run_id="route-01",
        window_handle_resolver=lambda title: calls.append(("window", title)) or 7,
        region_resolver=lambda handle: (
            calls.append(("region", handle)) or CaptureRegion(320, 156, 1920, 1080)
        ),
        window_probe_factory=lambda: _WindowProbe(),
        mss_factory=lambda region: calls.append(("source", region)) or source,
        gateway_factory=lambda: calls.append("gateway"),
    )

    assert runtime.target_window_handle == 7
    assert runtime.actuator is None
    assert "gateway" not in calls
    runtime.source.close()


def test_runtime_rejects_non_1080_client_before_source_or_gateway(tmp_path) -> None:
    calls: list[str] = []

    with pytest.raises(ValueError, match="1920x1080"):
        build_windows_route_capture_runtime(
            _settings(),
            artifacts=tmp_path / "run",
            armed=False,
            run_id="route-01",
            window_handle_resolver=lambda _title: 7,
            region_resolver=lambda _handle: CaptureRegion(0, 0, 1904, 1041),
            mss_factory=lambda _region: calls.append("source"),
            gateway_factory=lambda: calls.append("gateway"),
        )

    assert calls == []


class _WindowProbe:
    def foreground_window_handle(self) -> int:
        return 7

    def window_title(self, _handle: int) -> str:
        return "三角洲行动  "


def test_exact_window_guard_rejects_client_position_drift() -> None:
    regions = iter(
        [
            CaptureRegion(320, 156, 1920, 1080),
            CaptureRegion(321, 156, 1920, 1080),
        ]
    )
    guard = ExactWindowGuard(
        target_window_title="三角洲行动  ",
        target_window_handle=7,
        expected_region=CaptureRegion(320, 156, 1920, 1080),
        region_resolver=lambda _handle: next(regions),
        window_probe=_WindowProbe(),
        input_gate=None,
    )

    guard.check()
    with pytest.raises(RuntimeError, match="客户区发生变化"):
        guard.check()


class _WrongWindowProbe:
    def foreground_window_handle(self) -> int:
        return 8

    def window_title(self, _handle: int) -> str:
        return "三角洲行动  "


def test_exact_window_guard_rejects_different_same_title_handle() -> None:
    guard = ExactWindowGuard(
        target_window_title="三角洲行动  ",
        target_window_handle=7,
        expected_region=CaptureRegion(320, 156, 1920, 1080),
        region_resolver=lambda _handle: CaptureRegion(320, 156, 1920, 1080),
        window_probe=_WrongWindowProbe(),
        input_gate=None,
    )

    with pytest.raises(RuntimeError, match="精确匹配"):
        guard.check()


def test_dry_run_requires_three_in_match_frames_and_generates_zero_input(tmp_path) -> None:
    clock = _Clock()
    runtime = _runtime(
        tmp_path,
        frames=[_frame(0, 0), _frame(1, 10), _frame(2, 20)],
        clock=clock,
        actuator=None,
    )

    result = run_route_capture_loop(
        settings=_settings(),
        runtime=runtime,
        armed=False,
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )

    assert result.status is RouteCaptureStatus.DRY_RUN_VALIDATED
    assert result.actual_input_event_count == 0
    assert len(list(ReplayFrameSource(tmp_path / "run/hud"))) == 3
    assert not (tmp_path / "run/dataset/input-events.jsonl").exists()


def test_unknown_hud_frame_resets_initial_confirmation(tmp_path) -> None:
    clock = _Clock()
    runtime = _runtime(
        tmp_path,
        frames=[
            _frame(0, 0),
            _frame(1, 10),
            _frame(2, 20),
            _frame(3, 30),
            _frame(4, 40),
            _frame(5, 50),
        ],
        clock=clock,
        actuator=None,
        observer=_Observer(unknown_sequences={2}),
    )

    result = run_route_capture_loop(
        settings=_settings(),
        runtime=runtime,
        armed=False,
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )

    assert result.status is RouteCaptureStatus.DRY_RUN_VALIDATED
    assert result.hud_frame_count == 6


@pytest.mark.parametrize(
    "frames",
    [
        [_frame(0, 100), _frame(1, 90)],
        [_frame(0, 100), _frame(0, 110)],
    ],
)
def test_invalid_hud_frame_order_stops_before_input(tmp_path, frames) -> None:
    clock = _Clock()
    actuator = _Actuator(clock)
    runtime = _runtime(
        tmp_path,
        frames=frames,
        clock=clock,
        actuator=actuator,
    )

    with pytest.raises(RuntimeError, match="严格递增"):
        run_route_capture_loop(
            settings=_settings(),
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert actuator.events == ()
    assert runtime.source.closed is True


def test_armed_step_persists_pre_then_inputs_then_post(tmp_path) -> None:
    clock = _Clock()
    actuator = _Actuator(clock)
    runtime = _runtime(
        tmp_path,
        frames=[
            _frame(0, 0),
            _frame(1, 10),
            _frame(2, 20),
            _frame(3, 30),
            _frame(4, 200),
        ],
        clock=clock,
        actuator=actuator,
    )

    result = run_route_capture_loop(
        settings=_settings(),
        runtime=runtime,
        armed=True,
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )

    assert result.status is RouteCaptureStatus.COMPLETED
    assert result.route_frame_count == 2
    assert result.actual_input_event_count == 3
    assert actuator.pressed_keys == frozenset()
    replay = list(ReplayFrameSource(tmp_path / "run/dataset"))
    assert [frame.metadata["route_capture"]["phase"] for frame in replay] == [
        "before",
        "after",
    ]
    input_records = [
        json.loads(line)
        for line in (tmp_path / "run/dataset/input-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["payload"]["kind"] for record in input_records] == [
        "mouse_move",
        "key_down",
        "key_up",
    ]
    assert (tmp_path / "run/dataset/run.json").is_file()
    assert not (tmp_path / "run/dataset/partial-run.json").exists()


def test_post_frame_failure_is_partial_and_step_is_not_retried(tmp_path) -> None:
    clock = _Clock()
    actuator = _Actuator(clock)
    runtime = _runtime(
        tmp_path,
        frames=[
            _frame(0, 0),
            _frame(1, 10),
            _frame(2, 20),
            _frame(3, 30),
            _frame(4, 25),
        ],
        clock=clock,
        actuator=actuator,
    )

    with pytest.raises(RuntimeError, match="严格递增"):
        run_route_capture_loop(
            settings=_settings(),
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert [event.kind for event in actuator.events] == [
        "mouse_move",
        "key_down",
        "key_up",
    ]
    assert actuator.pressed_keys == frozenset()
    assert (tmp_path / "run/dataset/partial-run.json").is_file()
    assert not (tmp_path / "run/dataset/run.json").exists()


def test_pre_frame_write_failure_prevents_all_input(tmp_path) -> None:
    clock = _Clock()
    actuator = _Actuator(clock)
    runtime = _runtime(
        tmp_path,
        frames=[
            _frame(0, 0),
            _frame(1, 10),
            _frame(2, 20),
            _frame(3, 30),
        ],
        clock=clock,
        actuator=actuator,
    )
    runtime.dataset_recorder.record = lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("disk full")
    )

    with pytest.raises(OSError, match="disk full"):
        run_route_capture_loop(
            settings=_settings(),
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert actuator.events == ()
    assert actuator.pressed_keys == frozenset()


def test_f12_during_pulse_releases_key_and_skips_remaining_steps(tmp_path) -> None:
    clock = _Clock()
    actuator = _Actuator(clock)
    guard = _Guard()
    steps = (
        RouteCaptureStep("step-1", ("w",), 80, 0, 0, 10),
        RouteCaptureStep("step-2", ("d",), 80, 0, 0, 10),
    )
    settings = _settings(steps=steps)
    runtime = _runtime(
        tmp_path,
        frames=[
            _frame(0, 0),
            _frame(1, 10),
            _frame(2, 20),
            _frame(3, 30),
        ],
        clock=clock,
        actuator=actuator,
        guard=guard,
    )

    def interrupting_sleep(seconds: float) -> None:
        clock.sleep(seconds)
        guard.error = EmergencyStopError("F12")

    with pytest.raises(EmergencyStopError, match="F12"):
        run_route_capture_loop(
            settings=settings,
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=interrupting_sleep,
        )

    assert actuator.pressed_keys == frozenset()
    assert actuator.release_all_count >= 1
    assert [event.key for event in actuator.events if event.kind == "key_down"] == ["w"]
    assert any(event.kind == "key_up" and event.key == "w" for event in actuator.events)


def test_guarded_pulse_rejects_expired_pre_frame_before_input() -> None:
    clock = _Clock()
    clock.now_ns = 1_000_000_000
    actuator = _Actuator(clock)
    executor = GuardedPulseExecutor(
        actuator=actuator,
        guard=_Guard(),
        guard_interval_ms=20,
        maximum_frame_age_ms=500,
        persist_inputs=lambda: None,
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )

    with pytest.raises(RuntimeError, match="过期"):
        executor.execute(
            RouteCaptureStep("expired", ("w",), 40, 0, 0, 0),
            authorized_at_ns=0,
        )

    assert actuator.events == ()
