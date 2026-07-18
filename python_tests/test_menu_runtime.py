import threading

import pytest

from delta_vision.actuator import DryRunActuator
from delta_vision.config import CaptureRegion
from delta_vision.menu_automation import (
    MenuAction,
    MenuActionKind,
    MenuScene,
)
from delta_vision.menu_runtime import (
    DuplicateMenuActionError,
    ExpiredMenuActionError,
    MenuActionCleanupError,
    MenuActionExecutor,
    MenuExecutionStatus,
)


class FakeActuator:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.action_error: BaseException | None = None
        self.cleanup_error: BaseException | None = None
        self.entered_action = threading.Event()
        self.release_action = threading.Event()
        self.block_action = False

    def click_left_at(
        self,
        screen_x: int,
        screen_y: int,
        *,
        now_ns: int,
        expires_at_ns: int,
    ) -> None:
        self.calls.append(("click", screen_x, screen_y, now_ns, expires_at_ns))
        self.entered_action.set()
        if self.block_action:
            assert self.release_action.wait(timeout=1)
        if self.action_error is not None:
            raise self.action_error

    def tap_key(self, key: str, *, now_ns: int, expires_at_ns: int) -> None:
        self.calls.append(("tap", key, now_ns, expires_at_ns))
        if self.action_error is not None:
            raise self.action_error

    def release_all(self, *, now_ns: int, reason: str) -> None:
        self.calls.append(("release_all", now_ns, reason))
        if self.cleanup_error is not None:
            raise self.cleanup_error


def _click_action(
    *,
    position: tuple[int, int] = (20, 30),
    expires_at_ns: int = 200,
) -> MenuAction:
    return MenuAction(
        kind=MenuActionKind.CLICK,
        source=MenuScene.LOBBY,
        expected_target=MenuScene.STRATEGY_BOARD,
        position=position,
        expires_at_ns=expires_at_ns,
    )


def _key_action(*, expires_at_ns: int = 200) -> MenuAction:
    return MenuAction(
        kind=MenuActionKind.KEY,
        source=MenuScene.DEATH_SUMMARY,
        expected_target=MenuScene.LOBBY,
        key="space",
        expires_at_ns=expires_at_ns,
    )


def _executor(actuator: FakeActuator | DryRunActuator) -> MenuActionExecutor:
    return MenuActionExecutor(
        actuator=actuator,
        capture_region=CaptureRegion(left=-100, top=50, width=2560, height=1440),
        clock_ns=lambda: 0,
    )


def test_executor_converts_capture_local_click_to_virtual_screen_coordinates() -> None:
    actuator = FakeActuator()
    executor = _executor(actuator)

    record = executor.execute(_click_action(), now_ns=100)

    assert actuator.calls == [("click", -80, 80, 100, 200)]
    assert record.status is MenuExecutionStatus.SUCCEEDED
    assert record.local_position == (20, 30)
    assert record.screen_position == (-80, 80)
    assert record.error is None
    assert executor.records == (record,)


def test_executor_routes_key_action_to_atomic_tap() -> None:
    actuator = FakeActuator()
    executor = _executor(actuator)

    record = executor.execute(_key_action(), now_ns=100)

    assert actuator.calls == [("tap", "space", 100, 200)]
    assert record.status is MenuExecutionStatus.SUCCEEDED
    assert record.key == "space"
    assert record.screen_position is None


def test_executor_rejects_expired_action_before_actuator_call() -> None:
    actuator = FakeActuator()
    executor = _executor(actuator)

    with pytest.raises(ExpiredMenuActionError, match="过期"):
        executor.execute(_click_action(), now_ns=200)

    assert actuator.calls == []
    assert executor.records == ()


@pytest.mark.parametrize("position", [(-1, 0), (0, -1), (2560, 0), (0, 1440)])
def test_executor_rejects_click_outside_capture_region_without_input(
    position: tuple[int, int],
) -> None:
    actuator = FakeActuator()
    executor = _executor(actuator)

    with pytest.raises(ValueError, match="采集区域"):
        executor.execute(_click_action(position=position), now_ns=100)

    assert actuator.calls == []
    assert executor.records == ()


def test_executor_never_retries_same_transition_even_with_new_expiry() -> None:
    actuator = FakeActuator()
    executor = _executor(actuator)
    executor.execute(_click_action(), now_ns=100)

    with pytest.raises(DuplicateMenuActionError, match="重复"):
        executor.execute(_click_action(expires_at_ns=300), now_ns=150)

    assert [call[0] for call in actuator.calls] == ["click"]


def test_executor_failure_releases_all_and_consumes_transition() -> None:
    actuator = FakeActuator()
    actuator.action_error = RuntimeError("input failed")
    executor = _executor(actuator)

    with pytest.raises(RuntimeError, match="input failed"):
        executor.execute(_click_action(), now_ns=100)

    assert [call[0] for call in actuator.calls] == ["click", "release_all"]
    assert executor.records[-1].status is MenuExecutionStatus.FAILED
    assert executor.records[-1].error == "input failed"
    with pytest.raises(DuplicateMenuActionError):
        executor.execute(_click_action(expires_at_ns=300), now_ns=150)


def test_executor_keyboard_interrupt_still_releases_and_records_failure() -> None:
    actuator = FakeActuator()
    actuator.action_error = KeyboardInterrupt("stop")
    executor = _executor(actuator)

    with pytest.raises(KeyboardInterrupt, match="stop"):
        executor.execute(_click_action(), now_ns=100)

    assert [call[0] for call in actuator.calls] == ["click", "release_all"]
    assert executor.records[-1].status is MenuExecutionStatus.FAILED
    assert executor.records[-1].error == "stop"
    with pytest.raises(DuplicateMenuActionError):
        executor.execute(_click_action(expires_at_ns=300), now_ns=150)


def test_executor_reports_action_and_cleanup_failure_without_retry() -> None:
    actuator = FakeActuator()
    actuator.action_error = RuntimeError("input failed")
    actuator.cleanup_error = RuntimeError("cleanup failed")
    executor = _executor(actuator)

    with pytest.raises(MenuActionCleanupError, match="cleanup failed"):
        executor.execute(_click_action(), now_ns=100)

    record = executor.records[-1]
    assert record.status is MenuExecutionStatus.FAILED
    assert "input failed" in (record.error or "")
    assert "cleanup failed" in (record.error or "")


def test_executor_serializes_duplicate_transition_at_most_once() -> None:
    actuator = FakeActuator()
    actuator.block_action = True
    executor = _executor(actuator)
    errors: list[BaseException] = []

    def execute(action: MenuAction, now_ns: int) -> None:
        try:
            executor.execute(action, now_ns=now_ns)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=execute, args=(_click_action(), 100))
    second = threading.Thread(
        target=execute,
        args=(_click_action(expires_at_ns=300), 150),
    )
    first.start()
    assert actuator.entered_action.wait(timeout=1)
    second.start()
    actuator.release_action.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DuplicateMenuActionError)
    assert [call[0] for call in actuator.calls] == ["click"]


def test_executor_rechecks_different_transition_expiry_after_waiting_for_lock() -> None:
    actuator = FakeActuator()
    actuator.block_action = True
    clock_now = [100]
    executor = MenuActionExecutor(
        actuator=actuator,
        capture_region=CaptureRegion(left=0, top=0, width=2560, height=1440),
        clock_ns=lambda: clock_now[0],
    )
    second_started = threading.Event()
    errors: list[BaseException] = []

    def first_action() -> None:
        executor.execute(_click_action(expires_at_ns=500), now_ns=100)

    def stale_second_action() -> None:
        second_started.set()
        try:
            executor.execute(_key_action(expires_at_ns=200), now_ns=100)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=first_action)
    second = threading.Thread(target=stale_second_action)
    first.start()
    assert actuator.entered_action.wait(timeout=1)
    second.start()
    assert second_started.wait(timeout=1)
    clock_now[0] = 201
    actuator.release_action.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ExpiredMenuActionError)
    assert [call[0] for call in actuator.calls] == ["click"]
    assert len(executor.records) == 1

    # 过期发生在 actuator 尝试前，不消费该 transition；新鲜意图仍可执行一次。
    executor.execute(_key_action(expires_at_ns=300), now_ns=250)
    assert [call[0] for call in actuator.calls] == ["click", "tap"]
    assert len(executor.records) == 2


def test_executor_integrates_with_dry_run_without_os_input() -> None:
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)
    executor = _executor(actuator)

    executor.execute(_click_action(), now_ns=100)
    executor.execute(_key_action(expires_at_ns=300), now_ns=200)

    assert [event.kind for event in actuator.events] == [
        "mouse_move_absolute",
        "mouse_left_down",
        "mouse_left_up",
        "key_down",
        "key_up",
    ]
    assert actuator.pressed_keys == frozenset()
