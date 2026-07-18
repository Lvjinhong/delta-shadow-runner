"""把一次性菜单动作意图安全地交给 dry-run 或 Win32 actuator。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .config import CaptureRegion
from .menu_automation import MenuAction, MenuActionKind, MenuScene


class ExpiredMenuActionError(RuntimeError):
    """菜单动作在执行器获得串行锁前已经过期。"""


class DuplicateMenuActionError(RuntimeError):
    """同一页面转换已经尝试过，禁止自动重试。"""


class MenuActionCleanupError(RuntimeError):
    """菜单动作失败后，输入释放也失败。"""


class MenuExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MenuInputActuator(Protocol):
    def click_left_at(
        self,
        screen_x: int,
        screen_y: int,
        *,
        now_ns: int,
        expires_at_ns: int,
    ) -> None: ...

    def tap_key(self, key: str, *, now_ns: int, expires_at_ns: int) -> None: ...

    def release_all(self, *, now_ns: int, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class MenuExecutionRecord:
    status: MenuExecutionStatus
    source: MenuScene
    expected_target: MenuScene
    kind: MenuActionKind
    attempted_at_ns: int
    expires_at_ns: int
    local_position: tuple[int, int] | None
    screen_position: tuple[int, int] | None
    key: str | None
    error: str | None


class MenuActionExecutor:
    """每个页面转换最多尝试一次，并把局部坐标转换到虚拟桌面。"""

    def __init__(
        self,
        *,
        actuator: MenuInputActuator,
        capture_region: CaptureRegion,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._actuator = actuator
        self._capture_region = capture_region
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._consumed_transitions: set[tuple[MenuScene, MenuScene]] = set()
        self._records: list[MenuExecutionRecord] = []

    @property
    def records(self) -> tuple[MenuExecutionRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @staticmethod
    def _validate_time(*, now_ns: int, expires_at_ns: int) -> None:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("菜单动作时钟必须是非负整数纳秒")
        if now_ns >= expires_at_ns:
            raise ExpiredMenuActionError("菜单动作已经过期")

    def _screen_position(self, action: MenuAction) -> tuple[int, int] | None:
        if action.kind is not MenuActionKind.CLICK:
            return None
        if action.position is None:
            raise ValueError("点击菜单动作缺少局部坐标")
        local_x, local_y = action.position
        if not (
            0 <= local_x < self._capture_region.width
            and 0 <= local_y < self._capture_region.height
        ):
            raise ValueError("点击坐标不在采集区域内")
        return (
            self._capture_region.left + local_x,
            self._capture_region.top + local_y,
        )

    @staticmethod
    def _record(
        action: MenuAction,
        *,
        status: MenuExecutionStatus,
        attempted_at_ns: int,
        screen_position: tuple[int, int] | None,
        error: str | None,
    ) -> MenuExecutionRecord:
        return MenuExecutionRecord(
            status=status,
            source=action.source,
            expected_target=action.expected_target,
            kind=action.kind,
            attempted_at_ns=attempted_at_ns,
            expires_at_ns=action.expires_at_ns,
            local_position=action.position,
            screen_position=screen_position,
            key=action.key,
            error=error,
        )

    def execute(self, action: MenuAction, *, now_ns: int) -> MenuExecutionRecord:
        if not isinstance(action, MenuAction):
            raise TypeError("action 必须是 MenuAction")
        self._validate_time(now_ns=now_ns, expires_at_ns=action.expires_at_ns)
        screen_position = self._screen_position(action)
        transition_token = (action.source, action.expected_target)

        with self._lock:
            dispatch_now_ns = max(now_ns, self._clock_ns())
            self._validate_time(
                now_ns=dispatch_now_ns,
                expires_at_ns=action.expires_at_ns,
            )
            if transition_token in self._consumed_transitions:
                raise DuplicateMenuActionError(
                    f"页面转换禁止重复执行: {action.source} -> {action.expected_target}"
                )
            # 在调用 actuator 前即消费动作；部分输入失败后也绝不自动重试。
            self._consumed_transitions.add(transition_token)
            try:
                if action.kind is MenuActionKind.CLICK:
                    if screen_position is None:
                        raise ValueError("点击菜单动作缺少屏幕坐标")
                    self._actuator.click_left_at(
                        *screen_position,
                        now_ns=dispatch_now_ns,
                        expires_at_ns=action.expires_at_ns,
                    )
                elif action.kind is MenuActionKind.KEY:
                    if action.key is None:
                        raise ValueError("按键菜单动作缺少 key")
                    self._actuator.tap_key(
                        action.key,
                        now_ns=dispatch_now_ns,
                        expires_at_ns=action.expires_at_ns,
                    )
                else:
                    raise ValueError("不支持的菜单动作类型")
            except BaseException as action_error:
                try:
                    self._actuator.release_all(
                        now_ns=dispatch_now_ns,
                        reason=f"菜单动作执行失败: {action.source}",
                    )
                except BaseException as cleanup_error:
                    combined_error = (
                        f"动作失败: {action_error}; 清理失败: {cleanup_error}"
                    )
                    self._records.append(
                        self._record(
                            action,
                            status=MenuExecutionStatus.FAILED,
                            attempted_at_ns=dispatch_now_ns,
                            screen_position=screen_position,
                            error=combined_error,
                        )
                    )
                    raise MenuActionCleanupError(combined_error) from cleanup_error
                self._records.append(
                    self._record(
                        action,
                        status=MenuExecutionStatus.FAILED,
                        attempted_at_ns=dispatch_now_ns,
                        screen_position=screen_position,
                        error=str(action_error),
                    )
                )
                raise

            record = self._record(
                action,
                status=MenuExecutionStatus.SUCCEEDED,
                attempted_at_ns=dispatch_now_ns,
                screen_position=screen_position,
                error=None,
            )
            self._records.append(record)
            return record
