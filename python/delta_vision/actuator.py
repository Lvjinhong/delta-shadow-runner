"""不会触碰操作系统的 dry-run 输入执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ActionEvent:
    kind: Literal["key_down", "key_up"]
    key: str
    at_ns: int
    dry_run: bool
    reason: str | None = None


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

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._pressed_at)

    @property
    def events(self) -> tuple[ActionEvent, ...]:
        return tuple(self._events)

    def _require_allowed(self, key: str) -> None:
        if key not in self._allowed_keys:
            raise ValueError(f'不允许的按键: "{key}"')

    def key_down(self, key: str, *, now_ns: int) -> None:
        self._require_allowed(key)
        if key in self._pressed_at:
            return
        self._pressed_at[key] = now_ns
        self._events.append(ActionEvent("key_down", key, now_ns, dry_run=True))

    def key_up(self, key: str, *, now_ns: int, reason: str | None = None) -> None:
        self._require_allowed(key)
        if key not in self._pressed_at:
            return
        del self._pressed_at[key]
        self._events.append(ActionEvent("key_up", key, now_ns, dry_run=True, reason=reason))

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
