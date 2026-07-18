"""不会触碰操作系统的 dry-run 输入执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class ExpiredDryRunActionError(RuntimeError):
    """dry-run 视觉动作意图在记录前已经过期。"""


@dataclass(frozen=True, slots=True)
class ActionEvent:
    kind: Literal[
        "key_down",
        "key_up",
        "mouse_move",
        "mouse_move_absolute",
        "mouse_left_down",
        "mouse_left_up",
    ]
    key: str | None
    at_ns: int
    dry_run: bool
    reason: str | None = None
    dx: int | None = None
    dy: int | None = None
    x: int | None = None
    y: int | None = None


class DryRunActuator:
    """记录动作并强制按键生命周期，不发送真实输入。"""

    def __init__(self, *, allowed_keys: set[str], max_key_hold_ms: int) -> None:
        if not allowed_keys:
            raise ValueError("允许按键集合不能为空")
        if max_key_hold_ms <= 0:
            raise ValueError("最大按键时长必须为正数")
        self._allowed_keys = frozenset(allowed_keys)
        self._max_key_hold_ns = max_key_hold_ms * 1_000_000
        self._pressed_at: dict[str, int] = {}
        self._events: list[ActionEvent] = []
        self._last_event_at_ns = -1

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._pressed_at)

    @property
    def events(self) -> tuple[ActionEvent, ...]:
        return tuple(self._events)

    def _require_allowed(self, key: str) -> None:
        if key not in self._allowed_keys:
            raise ValueError(f'不允许的按键: "{key}"')

    def _next_event_time(self, now_ns: int) -> int:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("动作时间戳必须是非负整数")
        effective_now_ns = max(now_ns, self._last_event_at_ns)
        self._last_event_at_ns = effective_now_ns
        return effective_now_ns

    def key_down(self, key: str, *, now_ns: int) -> None:
        self._require_allowed(key)
        if key in self._pressed_at:
            return
        effective_now_ns = self._next_event_time(now_ns)
        self._pressed_at[key] = effective_now_ns
        self._events.append(ActionEvent("key_down", key, effective_now_ns, dry_run=True))

    def key_up(self, key: str, *, now_ns: int, reason: str | None = None) -> None:
        self._require_allowed(key)
        if key not in self._pressed_at:
            return
        del self._pressed_at[key]
        effective_now_ns = self._next_event_time(now_ns)
        self._events.append(
            ActionEvent("key_up", key, effective_now_ns, dry_run=True, reason=reason)
        )

    def move_mouse_relative(self, dx: int, dy: int, *, now_ns: int) -> None:
        if type(dx) is not int or type(dy) is not int:
            raise ValueError("相对鼠标位移必须是整数")
        effective_now_ns = self._next_event_time(now_ns)
        self._events.append(
            ActionEvent(
                "mouse_move",
                None,
                effective_now_ns,
                dry_run=True,
                dx=dx,
                dy=dy,
            )
        )

    def click_left_at(
        self,
        screen_x: int,
        screen_y: int,
        *,
        now_ns: int,
        expires_at_ns: int,
    ) -> None:
        if type(screen_x) is not int or type(screen_y) is not int:
            raise ValueError("点击屏幕坐标必须是整数")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("点击时间戳必须是非负整数")
        if type(expires_at_ns) is not int or expires_at_ns <= 0:
            raise ValueError("点击过期时间必须是正整数")
        effective_now_ns = max(now_ns, self._last_event_at_ns)
        if effective_now_ns >= expires_at_ns:
            raise ExpiredDryRunActionError("dry-run 视觉点击动作已经过期")

        effective_now_ns = self._next_event_time(effective_now_ns)
        self._events.extend(
            (
                ActionEvent(
                    "mouse_move_absolute",
                    None,
                    effective_now_ns,
                    dry_run=True,
                    x=screen_x,
                    y=screen_y,
                ),
                ActionEvent(
                    "mouse_left_down",
                    None,
                    effective_now_ns,
                    dry_run=True,
                ),
                ActionEvent(
                    "mouse_left_up",
                    None,
                    effective_now_ns,
                    dry_run=True,
                    reason="点击完成",
                ),
            )
        )

    def release_all(self, *, now_ns: int, reason: str) -> None:
        # 修饰键和移动键按按下顺序逆序释放，避免组合键残留。
        for key in reversed(tuple(self._pressed_at)):
            self.key_up(key, now_ns=now_ns, reason=reason)

    def expire_overdue(self, *, now_ns: int) -> tuple[str, ...]:
        expired = tuple(
            sorted(
                key
                for key, pressed_at_ns in self._pressed_at.items()
                if now_ns - pressed_at_ns >= self._max_key_hold_ns
            )
        )
        for key in expired:
            self.key_up(key, now_ns=now_ns, reason="超过最大按键时长")
        return expired
