import json
import threading
import time
from pathlib import Path

import pytest

from delta_vision import route_capture as route_capture_module
from delta_vision.config import CaptureRegion
from delta_vision.route_capture import (
    GuardedPulseExecutor,
    RouteCaptureStep,
    build_windows_route_capture_runtime,
    load_route_capture_settings,
    run_route_capture_loop,
)
from delta_vision.safe_input import EmergencyStopError, SafetyGate, Win32InputActuator
from test_route_capture import (
    _Actuator,
    _Clock,
    _frame,
    _Observer,
    _Profile,
    _runtime,
    _settings,
    _Source,
    _write_config,
)


def test_load_settings_requires_f12_as_emergency_stop(tmp_path) -> None:
    path = _write_config(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["emergency_virtual_key"] = 65
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="F12"):
        load_route_capture_settings(
            path,
            menu_profile_loader=lambda _path: _Profile(observer=_Observer()),
        )


def test_armed_runtime_caps_independent_watchdog_below_200ms(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    source = _Source([], _Clock())

    def actuator_factory(**kwargs):
        captured.update(kwargs)
        return _Actuator(_Clock())

    monkeypatch.setattr(
        "delta_vision.route_capture_windows.Win32InputActuator",
        actuator_factory,
    )
    runtime = build_windows_route_capture_runtime(
        _settings(),
        artifacts=tmp_path / "run",
        armed=True,
        run_id="route-01",
        window_handle_resolver=lambda _title: 7,
        region_resolver=lambda _handle: CaptureRegion(0, 0, 1920, 1080),
        mss_factory=lambda _region: source,
        gateway_factory=object,
    )

    assert captured["max_key_hold_ms"] == 150
    runtime.source.close()


class _PartialReleaseActuator(_Actuator):
    def __init__(self, clock: _Clock) -> None:
        super().__init__(clock)
        self._partial_release_attempted = False

    def release_all(self, *, now_ns: int, reason: str) -> None:
        self.release_all_count += 1
        if not self._partial_release_attempted:
            self._partial_release_attempted = True
            key = sorted(self._pressed)[0]
            self.key_up(key, now_ns=now_ns, reason="部分释放成功")
        raise RuntimeError("部分释放后失败")


def test_partial_release_events_are_persisted_even_when_release_raises(tmp_path) -> None:
    clock = _Clock()
    actuator = _PartialReleaseActuator(clock)
    settings = _settings(
        steps=(RouteCaptureStep("two-keys", ("w", "d"), 40, 0, 0, 0),)
    )
    runtime = _runtime(
        tmp_path,
        frames=[_frame(0, 0), _frame(1, 10), _frame(2, 20), _frame(3, 30)],
        clock=clock,
        actuator=actuator,
    )

    with pytest.raises(RuntimeError, match="部分释放后失败"):
        run_route_capture_loop(
            settings=settings,
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    actual_kinds = [event.kind for event in actuator.events]
    persisted_kinds = [
        json.loads(line)["payload"]["kind"]
        for line in (tmp_path / "run/dataset/input-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert actual_kinds == ["key_down", "key_down", "key_up"]
    assert persisted_kinds == actual_kinds
    assert not (tmp_path / "run/dataset/run.json").exists()
    assert (tmp_path / "run/dataset/partial-run.json").is_file()


def test_completed_marker_is_not_published_before_summary_succeeds(
    tmp_path, monkeypatch
) -> None:
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
    original_write = route_capture_module.write_atomic_json

    def fail_completed_summary(path, payload):
        if (
            Path(path).name == "route-capture-summary.json"
            and payload.get("status") == "completed"
        ):
            raise OSError("summary disk error")
        return original_write(path, payload)

    monkeypatch.setattr(route_capture_module, "write_atomic_json", fail_completed_summary)

    with pytest.raises(OSError, match="summary disk error"):
        run_route_capture_loop(
            settings=_settings(),
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert not (tmp_path / "run/dataset/run.json").exists()
    assert (tmp_path / "run/dataset/partial-run.json").is_file()


class _FinalCleanupFailsActuator(_Actuator):
    def release_all(self, *, now_ns: int, reason: str) -> None:
        if self.release_all_count:
            self.release_all_count += 1
            raise RuntimeError("最终输入清理失败")
        super().release_all(now_ns=now_ns, reason=reason)


def test_final_input_cleanup_failure_cannot_leave_completed_marker(tmp_path) -> None:
    clock = _Clock()
    actuator = _FinalCleanupFailsActuator(clock)
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

    with pytest.raises(RuntimeError, match="最终输入清理失败"):
        run_route_capture_loop(
            settings=_settings(),
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert not (tmp_path / "run/dataset/run.json").exists()
    assert (tmp_path / "run/dataset/partial-run.json").is_file()
    summary = json.loads(
        (tmp_path / "run/route-capture-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"


class _CloseFailsSource(_Source):
    def close(self) -> None:
        self.closed = True
        raise OSError("截图源关闭失败")


def test_source_close_failure_cannot_leave_completed_marker(tmp_path) -> None:
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
    runtime.source = _CloseFailsSource(
        [
            _frame(0, 0),
            _frame(1, 10),
            _frame(2, 20),
            _frame(3, 30),
            _frame(4, 200),
        ],
        clock,
    )

    with pytest.raises(OSError, match="截图源关闭失败"):
        run_route_capture_loop(
            settings=_settings(),
            runtime=runtime,
            armed=True,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert not (tmp_path / "run/dataset/run.json").exists()
    assert (tmp_path / "run/dataset/partial-run.json").is_file()
    summary = json.loads(
        (tmp_path / "run/route-capture-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"


class _Gateway:
    def __init__(self) -> None:
        self.emergency_pressed = False
        self.sent: list[tuple[int, bool]] = []

    def foreground_window_handle(self) -> int:
        return 7

    def foreground_title(self) -> str:
        return "三角洲行动  "

    def is_key_pressed(self, virtual_key: int) -> bool:
        assert virtual_key == 123
        return self.emergency_pressed

    def send_key(self, scan_code: int, *, key_up: bool) -> int:
        self.sent.append((scan_code, key_up))
        return 1

    def send_mouse_relative(self, dx: int, dy: int) -> int:
        raise AssertionError(f"不应发送鼠标输入: {dx}, {dy}")


class _ManualTimer:
    def __init__(self, interval_seconds: float, callback) -> None:
        self.interval_seconds = interval_seconds
        self._callback = callback

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    def fire(self) -> None:
        self._callback()


def test_blocked_persistence_cannot_prevent_independent_watchdog_release() -> None:
    gateway = _Gateway()
    timers: list[_ManualTimer] = []

    def timer_factory(interval_seconds, callback):
        timer = _ManualTimer(interval_seconds, callback)
        timers.append(timer)
        return timer

    gate = SafetyGate(
        target_window_title="三角洲行动  ",
        target_window_handle=7,
        emergency_virtual_key=123,
        gateway=gateway,
    )
    actuator = Win32InputActuator(
        scan_codes={"w": 0x11},
        max_key_hold_ms=150,
        gate=gate,
        gateway=gateway,
        timer_factory=timer_factory,
    )
    persist_started = threading.Event()
    allow_persist = threading.Event()
    persist_call_count = 0

    def blocking_persist() -> None:
        nonlocal persist_call_count
        persist_call_count += 1
        if persist_call_count == 1:
            persist_started.set()
            allow_persist.wait(timeout=1)

    executor = GuardedPulseExecutor(
        actuator=actuator,
        guard=gate,
        guard_interval_ms=20,
        maximum_frame_age_ms=1_000,
        persist_inputs=blocking_persist,
    )
    errors: list[BaseException] = []
    authorized_at_ns = time.monotonic_ns()
    worker = threading.Thread(
        target=lambda: _run_executor(executor, errors, authorized_at_ns),
        daemon=True,
    )
    worker.start()
    assert persist_started.wait(timeout=0.2)
    assert timers[0].interval_seconds == pytest.approx(0.15)
    gateway.emergency_pressed = True
    timers[0].fire()

    try:
        assert actuator.pressed_keys == frozenset()
    finally:
        allow_persist.set()
        worker.join(timeout=0.5)

    assert worker.is_alive() is False
    assert len(errors) == 1
    assert isinstance(errors[0], EmergencyStopError)
    assert gateway.sent == [(0x11, False), (0x11, True)]


def _run_executor(
    executor: GuardedPulseExecutor,
    errors: list[BaseException],
    authorized_at_ns: int,
) -> None:
    try:
        executor.execute(
            RouteCaptureStep("blocked-persist", ("w",), 80, 0, 0, 0),
            authorized_at_ns=authorized_at_ns,
        )
    except BaseException as error:
        errors.append(error)
