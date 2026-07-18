import threading

import pytest

from delta_vision.safe_input import (
    EmergencyStopError,
    ExpiredInputActionError,
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
        self.absolute_inserted_count = 1
        self.mouse_left_results: list[int] = []

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

    def send_mouse_absolute(self, screen_x: int, screen_y: int) -> int:
        self.sent.append(("mouse_absolute", screen_x, screen_y))
        return self.absolute_inserted_count

    def send_mouse_left(self, *, key_up: bool) -> int:
        self.sent.append(("mouse_left", key_up))
        if self.mouse_left_results:
            return self.mouse_left_results.pop(0)
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


def test_win32_actuator_taps_key_with_fresh_intent_and_releases() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    actuator.tap_key("w", now_ns=100, expires_at_ns=200)

    assert gateway.sent == [("key", 0x11, False), ("key", 0x11, True)]
    assert actuator.pressed_keys == frozenset()
    assert [event.kind for event in actuator.events] == ["key_down", "key_up"]
    assert actuator.events[-1].reason == "按键点击完成"


def test_win32_actuator_rejects_expired_key_tap_without_input() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 201)

    with pytest.raises(ExpiredInputActionError, match="过期"):
        actuator.tap_key("w", now_ns=100, expires_at_ns=200)

    assert gateway.sent == []


def test_win32_actuator_rechecks_key_tap_expiry_after_waiting_for_key_lock() -> None:
    gateway = FakeGateway()
    clock_now = [150]
    actuator = _actuator(gateway, clock_ns=lambda: clock_now[0])
    started = threading.Event()
    errors: list[BaseException] = []

    def tap_after_lock() -> None:
        started.set()
        try:
            actuator.tap_key("w", now_ns=100, expires_at_ns=200)
        except BaseException as error:
            errors.append(error)

    with actuator._key_locks["w"]:
        thread = threading.Thread(target=tap_after_lock)
        thread.start()
        assert started.wait(timeout=1)
        clock_now[0] = 201

    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ExpiredInputActionError)
    assert gateway.sent == []


def test_win32_actuator_rechecks_key_tap_expiry_immediately_before_down() -> None:
    clock_now = [150]

    class SlowGateGateway(FakeGateway):
        def foreground_title(self) -> str:
            title = super().foreground_title()
            clock_now[0] = 201
            return title

    gateway = SlowGateGateway()
    actuator = _actuator(gateway, clock_ns=lambda: clock_now[0])

    with pytest.raises(ExpiredInputActionError, match="过期"):
        actuator.tap_key("w", now_ns=100, expires_at_ns=200)

    assert gateway.sent == []
    assert actuator.pressed_keys == frozenset()


def test_win32_actuator_rejects_key_tap_when_key_is_already_pressed() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)
    actuator.key_down("w", now_ns=50)
    sent_before_tap = list(gateway.sent)

    with pytest.raises(InputInjectionError, match="尚未释放"):
        actuator.tap_key("w", now_ns=100, expires_at_ns=200)

    assert gateway.sent == sent_before_tap
    assert actuator.pressed_keys == frozenset({"w"})
    actuator.release_all(now_ns=160, reason="测试清理")


def test_win32_actuator_key_tap_always_attempts_release_after_key_up_failure() -> None:
    class FirstKeyUpFailingGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__()
            self.key_up_attempts = 0

        def send_key(self, scan_code: int, *, key_up: bool) -> int:
            self.sent.append(("key", scan_code, key_up))
            if key_up:
                self.key_up_attempts += 1
                return 0 if self.key_up_attempts == 1 else 1
            return 1

    gateway = FirstKeyUpFailingGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    with pytest.raises(InputInjectionError):
        actuator.tap_key("w", now_ns=100, expires_at_ns=200)

    assert gateway.sent == [
        ("key", 0x11, False),
        ("key", 0x11, True),
        ("key", 0x11, True),
    ]
    assert actuator.pressed_keys == frozenset()


def test_concurrent_key_taps_do_not_deadlock_emergency_release_all() -> None:
    class BarrierEmergencyGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__()
            self.emergency_barrier = threading.Barrier(2)
            self.w_gate_done = threading.Event()
            self.allow_a_return = threading.Event()

        def is_key_pressed(self, virtual_key: int) -> bool:
            assert virtual_key == 0x7B
            if not self.emergency_pressed:
                return False
            try:
                self.emergency_barrier.wait(timeout=0.1)
            except threading.BrokenBarrierError:
                pass
            if threading.current_thread().name == "tap-w":
                self.w_gate_done.set()
            else:
                assert self.allow_a_return.wait(timeout=1)
            return True

    gateway = BarrierEmergencyGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)
    actuator.key_down("w", now_ns=10)
    actuator.key_down("a", now_ns=20)
    gateway.emergency_pressed = True
    errors: list[BaseException] = []

    def tap(key: str) -> None:
        try:
            actuator.tap_key(key, now_ns=100, expires_at_ns=200)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=tap, args=("w",), name="tap-w", daemon=True)
    second = threading.Thread(target=tap, args=("a",), name="tap-a", daemon=True)
    first.start()
    second.start()
    assert gateway.w_gate_done.wait(timeout=1)
    # 让 tap-w 先进入 release_all 并阻塞在 tap-a 持有的 a 锁。
    assert not threading.Event().wait(timeout=0.05)
    gateway.allow_a_return.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(errors) == 2
    assert all(isinstance(error, EmergencyStopError) for error in errors)
    assert actuator.pressed_keys == frozenset()


def test_win32_actuator_keeps_event_time_monotonic_after_watchdog_race() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)
    actuator.key_down("w", now_ns=0)
    actuator.key_up("w", now_ns=300_000_000)

    actuator.key_down("d", now_ns=50_000_000)

    assert [event.at_ns for event in actuator.events] == [0, 300_000_000, 300_000_000]


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


def test_win32_actuator_clicks_at_absolute_position_and_releases_button() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    actuator.click_left_at(
        500,
        600,
        now_ns=100,
        expires_at_ns=200,
    )

    assert gateway.sent == [
        ("mouse_absolute", 500, 600),
        ("mouse_left", False),
        ("mouse_left", True),
    ]
    assert actuator.mouse_left_pressed is False
    assert [event.kind for event in actuator.events] == [
        "mouse_move_absolute",
        "mouse_left_down",
        "mouse_left_up",
    ]
    assert actuator.events[0].x == 500
    assert actuator.events[0].y == 600


def test_win32_actuator_rejects_expired_click_before_any_input() -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 200)

    with pytest.raises(ExpiredInputActionError, match="过期"):
        actuator.click_left_at(500, 600, now_ns=200, expires_at_ns=200)

    assert gateway.sent == []
    assert actuator.mouse_left_pressed is False


def test_win32_actuator_rechecks_expiry_after_pointer_move() -> None:
    gateway = FakeGateway()
    clock_values = iter((150, 201))
    actuator = _actuator(gateway, clock_ns=lambda: next(clock_values))

    with pytest.raises(ExpiredInputActionError, match="过期"):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert gateway.sent == [("mouse_absolute", 500, 600)]
    assert actuator.mouse_left_pressed is False


def test_win32_actuator_does_not_move_while_previous_left_down_is_stuck() -> None:
    gateway = FakeGateway()
    gateway.mouse_left_results = [1, 0, 0]
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    with pytest.raises(InputInjectionError):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)
    sent_before_retry = list(gateway.sent)

    with pytest.raises(InputInjectionError, match="尚未释放"):
        actuator.click_left_at(700, 800, now_ns=100, expires_at_ns=200)

    assert gateway.sent == sent_before_retry
    assert ("mouse_absolute", 700, 800) not in gateway.sent


def test_win32_actuator_rechecks_expiry_after_waiting_for_mouse_lock() -> None:
    gateway = FakeGateway()
    clock_now = [150]
    actuator = _actuator(gateway, clock_ns=lambda: clock_now[0])
    started = threading.Event()
    errors: list[BaseException] = []

    def click_after_lock() -> None:
        started.set()
        try:
            actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)
        except BaseException as error:
            errors.append(error)

    # 精确复现点击事务等待锁的竞态。
    with actuator._mouse_lock:
        thread = threading.Thread(target=click_after_lock)
        thread.start()
        assert started.wait(timeout=1)
        clock_now[0] = 201

    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ExpiredInputActionError)
    assert gateway.sent == []


def test_win32_actuator_serializes_absolute_move_and_click_transaction() -> None:
    class BlockingAbsoluteGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__()
            self.first_moved = threading.Event()
            self.release_first = threading.Event()
            self.second_moved = threading.Event()

        def send_mouse_absolute(self, screen_x: int, screen_y: int) -> int:
            inserted = super().send_mouse_absolute(screen_x, screen_y)
            if (screen_x, screen_y) == (100, 200):
                self.first_moved.set()
                assert self.release_first.wait(timeout=1)
            else:
                self.second_moved.set()
            return inserted

    gateway = BlockingAbsoluteGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)
    errors: list[BaseException] = []

    def click(x: int, y: int) -> None:
        try:
            actuator.click_left_at(x, y, now_ns=100, expires_at_ns=200)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=click, args=(100, 200))
    second = threading.Thread(target=click, args=(300, 400))
    first.start()
    assert gateway.first_moved.wait(timeout=1)
    second.start()
    assert not gateway.second_moved.wait(timeout=0.05)
    gateway.release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not errors
    assert not first.is_alive()
    assert not second.is_alive()
    assert gateway.sent == [
        ("mouse_absolute", 100, 200),
        ("mouse_left", False),
        ("mouse_left", True),
        ("mouse_absolute", 300, 400),
        ("mouse_left", False),
        ("mouse_left", True),
    ]


def test_relative_move_waits_until_absolute_click_transaction_finishes() -> None:
    class BlockingAbsoluteGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__()
            self.absolute_moved = threading.Event()
            self.release_absolute = threading.Event()
            self.relative_moved = threading.Event()

        def send_mouse_absolute(self, screen_x: int, screen_y: int) -> int:
            inserted = super().send_mouse_absolute(screen_x, screen_y)
            self.absolute_moved.set()
            assert self.release_absolute.wait(timeout=1)
            return inserted

        def send_mouse_relative(self, dx: int, dy: int) -> int:
            inserted = super().send_mouse_relative(dx, dy)
            self.relative_moved.set()
            return inserted

    gateway = BlockingAbsoluteGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)
    errors: list[BaseException] = []

    def click() -> None:
        try:
            actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)
        except BaseException as error:
            errors.append(error)

    def move() -> None:
        try:
            actuator.move_mouse_relative(25, 0, now_ns=160)
        except BaseException as error:
            errors.append(error)

    click_thread = threading.Thread(target=click)
    move_thread = threading.Thread(target=move)
    click_thread.start()
    assert gateway.absolute_moved.wait(timeout=1)
    move_thread.start()
    assert not gateway.relative_moved.wait(timeout=0.05)
    gateway.release_absolute.set()
    click_thread.join(timeout=1)
    move_thread.join(timeout=1)

    assert not errors
    assert not click_thread.is_alive()
    assert not move_thread.is_alive()
    assert gateway.sent == [
        ("mouse_absolute", 500, 600),
        ("mouse_left", False),
        ("mouse_left", True),
        ("mouse", 25, 0),
    ]


def test_win32_actuator_rechecks_foreground_after_pointer_move() -> None:
    class FocusLosingGateway(FakeGateway):
        def send_mouse_absolute(self, screen_x: int, screen_y: int) -> int:
            inserted = super().send_mouse_absolute(screen_x, screen_y)
            self.title = "Other Window"
            return inserted

    gateway = FocusLosingGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    with pytest.raises(ForegroundWindowError):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert gateway.sent == [("mouse_absolute", 500, 600)]
    assert actuator.mouse_left_pressed is False


def test_win32_actuator_always_releases_mouse_after_focus_changes_on_down() -> None:
    class FocusLosingGateway(FakeGateway):
        def send_mouse_left(self, *, key_up: bool) -> int:
            inserted = super().send_mouse_left(key_up=key_up)
            if not key_up:
                self.title = "Other Window"
            return inserted

    gateway = FocusLosingGateway()
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert gateway.sent[-2:] == [("mouse_left", False), ("mouse_left", True)]
    assert actuator.mouse_left_pressed is False


def test_win32_actuator_recovers_mouse_up_after_first_release_failure() -> None:
    gateway = FakeGateway()
    gateway.mouse_left_results = [1, 0, 1]
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    with pytest.raises(InputInjectionError, match="SendInput"):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert gateway.sent[-3:] == [
        ("mouse_left", False),
        ("mouse_left", True),
        ("mouse_left", True),
    ]
    assert actuator.mouse_left_pressed is False


def test_release_all_retries_mouse_up_after_click_release_failures() -> None:
    gateway = FakeGateway()
    gateway.mouse_left_results = [1, 0, 0, 1]
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    with pytest.raises(InputInjectionError):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)
    assert actuator.mouse_left_pressed is True

    actuator.release_all(now_ns=160, reason="测试清理")

    assert actuator.mouse_left_pressed is False
    assert actuator.events[-1].kind == "mouse_left_up"
    assert actuator.events[-1].reason == "测试清理"


def test_win32_actuator_does_not_press_mouse_when_absolute_move_fails() -> None:
    gateway = FakeGateway()
    gateway.absolute_inserted_count = 0
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    with pytest.raises(InputInjectionError):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert gateway.sent == [("mouse_absolute", 500, 600)]
    assert actuator.mouse_left_pressed is False


def test_win32_actuator_does_not_track_mouse_when_left_down_fails() -> None:
    gateway = FakeGateway()
    gateway.mouse_left_results = [0]
    actuator = _actuator(gateway, clock_ns=lambda: 150)

    with pytest.raises(InputInjectionError):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert gateway.sent[-1] == ("mouse_left", False)
    assert actuator.mouse_left_pressed is False


def test_mouse_watchdog_releases_button_without_control_loop_progress() -> None:
    gateway = FakeGateway()
    gateway.mouse_left_results = [1, 0, 0, 1]
    timers = FakeTimerFactory()
    actuator = _actuator(
        gateway,
        clock_ns=lambda: 351_000_000,
        timer_factory=timers,
    )

    with pytest.raises(InputInjectionError):
        actuator.click_left_at(
            500,
            600,
            now_ns=100_000_000,
            expires_at_ns=500_000_000,
        )
    assert actuator.mouse_left_pressed is True
    assert timers.timers[0].started is True

    timers.timers[0].fire()

    assert actuator.mouse_left_pressed is False
    assert actuator.events[-1].reason == "看门狗超过最大鼠标按下时长"


def test_mouse_watchdog_start_failure_releases_without_poisoning_next_action() -> None:
    class FailingStartTimer(FakeTimer):
        def start(self) -> None:
            raise RuntimeError("timer start failed")

    class FailingStartTimerFactory:
        def __call__(self, interval_seconds: float, callback) -> FailingStartTimer:
            return FailingStartTimer(interval_seconds, callback)

    gateway = FakeGateway()
    actuator = _actuator(
        gateway,
        clock_ns=lambda: 150,
        timer_factory=FailingStartTimerFactory(),  # type: ignore[arg-type]
    )

    with pytest.raises(InputInjectionError, match="看门狗"):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert actuator.mouse_left_pressed is False
    actuator.move_mouse_relative(1, 2, now_ns=160)
    assert gateway.sent[-1] == ("mouse", 1, 2)


def test_mouse_watchdog_factory_failure_immediately_releases_button() -> None:
    class FailingTimerFactory:
        def __call__(self, interval_seconds: float, callback) -> FakeTimer:
            raise RuntimeError("timer factory failed")

    gateway = FakeGateway()
    actuator = _actuator(
        gateway,
        clock_ns=lambda: 150,
        timer_factory=FailingTimerFactory(),  # type: ignore[arg-type]
    )

    with pytest.raises(InputInjectionError, match="看门狗"):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert gateway.sent == [
        ("mouse_absolute", 500, 600),
        ("mouse_left", False),
        ("mouse_left", True),
    ]
    assert actuator.mouse_left_pressed is False


def test_mouse_watchdog_start_and_immediate_release_failure_stays_recoverable() -> None:
    class FailingStartTimer(FakeTimer):
        def start(self) -> None:
            raise RuntimeError("timer start failed")

    class FailingStartTimerFactory:
        def __call__(self, interval_seconds: float, callback) -> FailingStartTimer:
            return FailingStartTimer(interval_seconds, callback)

    gateway = FakeGateway()
    gateway.mouse_left_results = [1, 0]
    actuator = _actuator(
        gateway,
        clock_ns=lambda: 150,
        timer_factory=FailingStartTimerFactory(),  # type: ignore[arg-type]
    )

    with pytest.raises(InputInjectionError, match="立即释放也失败"):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert actuator.mouse_left_pressed is True
    actuator.release_all(now_ns=160, reason="测试清理")
    assert actuator.mouse_left_pressed is False


def test_release_all_attempts_mouse_after_keyboard_release_failure() -> None:
    class KeyboardReleaseFailingGateway(FakeGateway):
        def send_key(self, scan_code: int, *, key_up: bool) -> int:
            self.sent.append(("key", scan_code, key_up))
            return 0 if key_up else 1

    gateway = KeyboardReleaseFailingGateway()
    gateway.mouse_left_results = [1, 0, 0, 1]
    actuator = _actuator(gateway, clock_ns=lambda: 150)
    actuator.key_down("w", now_ns=1)
    with pytest.raises(InputInjectionError):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    with pytest.raises(InputInjectionError):
        actuator.release_all(now_ns=160, reason="异常清理")

    assert actuator.mouse_left_pressed is False
    assert ("mouse_left", True) in gateway.sent


@pytest.mark.parametrize(
    ("now_ns", "expires_at_ns"),
    [(True, 200), (100, True), (-1, 200), (100, 0), (200, 200)],
)
def test_win32_actuator_rejects_invalid_or_expired_click_timing_without_input(
    now_ns: object,
    expires_at_ns: object,
) -> None:
    gateway = FakeGateway()
    actuator = _actuator(gateway)

    with pytest.raises((ValueError, ExpiredInputActionError)):
        actuator.click_left_at(
            500,
            600,
            now_ns=now_ns,  # type: ignore[arg-type]
            expires_at_ns=expires_at_ns,  # type: ignore[arg-type]
        )

    assert gateway.sent == []


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
