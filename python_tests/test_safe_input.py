import threading

import pytest

from delta_vision.safe_input import (
    EmergencyStopError,
    ForegroundWindowError,
    InputInjectionError,
    SafetyGate,
    Win32InputActuator,
)


class FakeGateway:
    def __init__(self) -> None:
        self.title = "Delta Vision Test Target"
        self.window_handle = 123
        self.emergency_pressed = False
        self.sent = []
        self.inserted_count = 1

    def foreground_title(self) -> str:
        return self.title

    def foreground_window_handle(self) -> int:
        return self.window_handle

    def is_key_pressed(self, virtual_key: int) -> bool:
        assert virtual_key == 0x7B
        return self.emergency_pressed

    def send_key(self, scan_code: int, *, key_up: bool) -> int:
        self.sent.append(("key", scan_code, key_up))
        return self.inserted_count

    def send_mouse_relative(self, dx: int, dy: int) -> int:
        self.sent.append(("mouse", dx, dy))
        return self.inserted_count


class FakeTimer:
    def __init__(self, interval_seconds: float, callback) -> None:
        self.interval_seconds = interval_seconds
        self._callback = callback
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self._callback()


class FakeTimerFactory:
    def __init__(self) -> None:
        self.timers: list[FakeTimer] = []

    def __call__(self, interval_seconds: float, callback) -> FakeTimer:
        timer = FakeTimer(interval_seconds, callback)
        self.timers.append(timer)
        return timer


def _actuator(
    gateway: FakeGateway,
    *,
    timer_factory: FakeTimerFactory | None = None,
    clock_ns=lambda: 999_000_000,
) -> Win32InputActuator:
    gate = SafetyGate(
        target_window_title="Delta Vision Test Target",
        target_window_handle=123,
        emergency_virtual_key=0x7B,
        gateway=gateway,
    )
    return Win32InputActuator(
        scan_codes={"w": 0x11, "a": 0x1E, "d": 0x20},
        max_key_hold_ms=250,
        gate=gate,
        gateway=gateway,
        clock_ns=clock_ns,
        timer_factory=timer_factory or FakeTimerFactory(),
    )


def test_win32_actuator_sends_scan_code_and_tracks_pressed_key() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)

    actuator.key_down("w", now_ns=1)
    actuator.key_down("w", now_ns=2)
    actuator.key_up("w", now_ns=3, reason="到达节点")

    assert gateway.sent == [("key", 0x11, False), ("key", 0x11, True)]
    assert actuator.pressed_keys == frozenset()
    assert actuator.events[-1].reason == "到达节点"


def test_win32_actuator_rejects_wrong_foreground_without_sending_input() -> None:
    gateway = FakeGateway()
    gateway.title = "Other Window"
    actuator = _actuator(gateway)

    with pytest.raises(ForegroundWindowError, match="前台窗口"):
        actuator.key_down("w", now_ns=1)

    assert gateway.sent == []
    assert actuator.pressed_keys == frozenset()


def test_win32_actuator_releases_pressed_keys_when_focus_is_lost() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)
    actuator.key_down("w", now_ns=1)
    gateway.title = "Other Window"

    with pytest.raises(ForegroundWindowError):
        actuator.key_down("a", now_ns=2)

    assert gateway.sent == [("key", 0x11, False), ("key", 0x11, True)]
    assert actuator.pressed_keys == frozenset()


def test_win32_actuator_releases_pressed_keys_on_emergency_stop() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)
    actuator.key_down("w", now_ns=1)
    gateway.emergency_pressed = True

    with pytest.raises(EmergencyStopError, match="急停"):
        actuator.move_mouse_relative(5, -3, now_ns=2)

    assert gateway.sent == [("key", 0x11, False), ("key", 0x11, True)]
    assert actuator.pressed_keys == frozenset()


def test_repeated_key_down_still_checks_emergency_stop() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)
    actuator.key_down("w", now_ns=1)
    gateway.emergency_pressed = True

    with pytest.raises(EmergencyStopError):
        actuator.key_down("w", now_ns=2)

    assert gateway.sent == [("key", 0x11, False), ("key", 0x11, True)]
    assert actuator.pressed_keys == frozenset()


def test_same_title_with_different_window_handle_is_rejected() -> None:
    gateway = FakeGateway()
    gateway.window_handle = 999
    actuator = _actuator(gateway)

    with pytest.raises(ForegroundWindowError, match="窗口句柄"):
        actuator.key_down("w", now_ns=1)

    assert gateway.sent == []


def test_win32_actuator_requires_every_event_to_be_inserted() -> None:
    gateway = FakeGateway()
    gateway.inserted_count = 0
    actuator = _actuator(gateway)

    with pytest.raises(InputInjectionError, match="SendInput"):
        actuator.key_down("w", now_ns=1)

    assert actuator.pressed_keys == frozenset()


def test_win32_actuator_sends_relative_mouse_motion() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)

    actuator.move_mouse_relative(12, -7, now_ns=1)

    assert gateway.sent == [("mouse", 12, -7)]


def test_win32_actuator_rechecks_gate_between_mouse_and_key_steps() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)
    actuator.move_mouse_relative(12, -7, now_ns=1)
    gateway.title = "Other Window"

    with pytest.raises(ForegroundWindowError):
        actuator.key_down("w", now_ns=2)

    assert gateway.sent == [("mouse", 12, -7)]
    assert actuator.pressed_keys == frozenset()


def test_win32_actuator_expires_overdue_keys() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)
    actuator.key_down("d", now_ns=1_000_000_000)

    expired = actuator.expire_overdue(now_ns=1_250_000_000)

    assert expired == ("d",)
    assert gateway.sent[-1] == ("key", 0x20, True)
    assert actuator.pressed_keys == frozenset()


def test_watchdog_releases_key_without_waiting_for_control_loop() -> None:
    gateway = FakeGateway()
    timers = FakeTimerFactory()
    actuator = _actuator(
        gateway,
        timer_factory=timers,
        clock_ns=lambda: 251_000_000,
    )

    actuator.key_down("w", now_ns=1_000_000)

    assert len(timers.timers) == 1
    assert timers.timers[0].interval_seconds == pytest.approx(0.25)
    assert timers.timers[0].started is True

    timers.timers[0].fire()

    assert gateway.sent == [("key", 0x11, False), ("key", 0x11, True)]
    assert actuator.pressed_keys == frozenset()
    assert actuator.events[-1].reason == "看门狗超过最大按键时长"


def test_blocked_foreground_check_does_not_block_watchdog_release() -> None:
    class BlockingGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__()
            self.should_block = False
            self.check_started = threading.Event()
            self.allow_check = threading.Event()

        def foreground_window_handle(self) -> int:
            if self.should_block:
                self.check_started.set()
                self.allow_check.wait(timeout=1)
            return super().foreground_window_handle()

    gateway = BlockingGateway()
    timers = FakeTimerFactory()
    actuator = _actuator(gateway, timer_factory=timers)
    actuator.key_down("w", now_ns=1)
    gateway.should_block = True
    action_thread = threading.Thread(
        target=lambda: actuator.key_down("a", now_ns=2),
        daemon=True,
    )
    action_thread.start()
    assert gateway.check_started.wait(timeout=0.2)
    watchdog_finished = threading.Event()
    watchdog_thread = threading.Thread(
        target=lambda: (timers.timers[0].fire(), watchdog_finished.set()),
        daemon=True,
    )
    watchdog_thread.start()

    try:
        assert watchdog_finished.wait(timeout=0.1)
        assert "w" not in actuator.pressed_keys
    finally:
        gateway.allow_check.set()
        action_thread.join(timeout=0.5)
        watchdog_thread.join(timeout=0.5)


def test_watchdog_retries_release_without_control_loop_progress() -> None:
    class RetryGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__()
            self.release_attempts = 0

        def send_key(self, scan_code: int, *, key_up: bool) -> int:
            self.sent.append(("key", scan_code, key_up))
            if key_up:
                self.release_attempts += 1
                return 0 if self.release_attempts == 1 else 1
            return 1

    gateway = RetryGateway()
    timers = FakeTimerFactory()
    actuator = _actuator(gateway, timer_factory=timers)
    actuator.key_down("w", now_ns=1)

    timers.timers[0].fire()

    assert actuator.pressed_keys == frozenset({"w"})
    assert len(timers.timers) == 2
    assert timers.timers[1].started is True

    timers.timers[1].fire()

    assert gateway.release_attempts == 2
    assert actuator.pressed_keys == frozenset()


def test_stale_watchdog_cannot_release_new_press_of_same_key() -> None:
    gateway = FakeGateway()
    timers = FakeTimerFactory()
    actuator = _actuator(gateway, timer_factory=timers)
    actuator.key_down("w", now_ns=1)
    stale_timer = timers.timers[0]
    actuator.key_up("w", now_ns=2)
    actuator.key_down("w", now_ns=3)

    stale_timer.fire()

    assert actuator.pressed_keys == frozenset({"w"})
    assert gateway.sent == [
        ("key", 0x11, False),
        ("key", 0x11, True),
        ("key", 0x11, False),
    ]


def test_expire_overdue_attempts_every_release_after_one_failure() -> None:
    class FailingReleaseGateway(FakeGateway):
        def send_key(self, scan_code: int, *, key_up: bool) -> int:
            self.sent.append(("key", scan_code, key_up))
            if key_up and scan_code == 0x11:
                return 0
            return 1

    gateway = FailingReleaseGateway()
    actuator = _actuator(gateway)
    actuator.key_down("w", now_ns=0)
    actuator.key_down("a", now_ns=0)

    with pytest.raises(InputInjectionError):
        actuator.expire_overdue(now_ns=250_000_000)

    assert ("key", 0x1E, True) in gateway.sent
    assert ("key", 0x11, True) in gateway.sent
    assert actuator.pressed_keys == frozenset({"w"})
