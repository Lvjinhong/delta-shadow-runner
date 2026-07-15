"""带前台窗口和急停保护的标准 Win32 输入执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


class ForegroundWindowError(RuntimeError):
    """目标窗口不是当前前台窗口。"""


class EmergencyStopError(RuntimeError):
    """用户按下急停键。"""


class InputInjectionError(RuntimeError):
    """Win32 SendInput 没有完整插入一个输入事件。"""


class InputGateway(Protocol):
    def foreground_window_handle(self) -> int: ...

    def foreground_title(self) -> str: ...

    def is_key_pressed(self, virtual_key: int) -> bool: ...

    def send_key(self, scan_code: int, *, key_up: bool) -> int: ...

    def send_mouse_relative(self, dx: int, dy: int) -> int: ...


@dataclass(frozen=True, slots=True)
class InputEvent:
    kind: Literal["key_down", "key_up", "mouse_move"]
    at_ns: int
    key: str | None = None
    dx: int | None = None
    dy: int | None = None
    reason: str | None = None


class SafetyGate:
    """在每个新的输入动作前验证急停键和前台窗口。"""

    def __init__(
        self,
        *,
        target_window_title: str,
        target_window_handle: int,
        emergency_virtual_key: int,
        gateway: InputGateway,
    ) -> None:
        if not target_window_title:
            raise ValueError("目标窗口标题不能为空")
        if target_window_handle <= 0:
            raise ValueError("目标窗口句柄必须为正数")
        self._target_window_title = target_window_title
        self._target_window_handle = target_window_handle
        self._emergency_virtual_key = emergency_virtual_key
        self._gateway = gateway

    def check(self) -> None:
        if self._gateway.is_key_pressed(self._emergency_virtual_key):
            raise EmergencyStopError("检测到急停键，已阻止输入")
        actual_handle = self._gateway.foreground_window_handle()
        if actual_handle != self._target_window_handle:
            raise ForegroundWindowError(
                f"前台窗口句柄不是目标窗口: 期望 {self._target_window_handle}，"
                f"实际 {actual_handle}"
            )
        actual_title = self._gateway.foreground_title()
        if actual_title != self._target_window_title:
            raise ForegroundWindowError(
                f'前台窗口不是目标窗口: 期望 "{self._target_window_title}"，实际 "{actual_title}"'
            )


class Win32InputActuator:
    """通过可注入 gateway 发送 scan code，并保证按键最终释放。"""

    def __init__(
        self,
        *,
        scan_codes: dict[str, int],
        max_key_hold_ms: int,
        gate: SafetyGate,
        gateway: InputGateway,
    ) -> None:
        if not scan_codes:
            raise ValueError("scan code 映射不能为空")
        if isinstance(max_key_hold_ms, bool) or max_key_hold_ms <= 0:
            raise ValueError("最大按键时长必须为正数")
        self._scan_codes = dict(scan_codes)
        self._max_key_hold_ns = max_key_hold_ms * 1_000_000
        self._gate = gate
        self._gateway = gateway
        self._pressed_at: dict[str, int] = {}
        self._events: list[InputEvent] = []

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._pressed_at)

    @property
    def events(self) -> tuple[InputEvent, ...]:
        return tuple(self._events)

    def _scan_code(self, key: str) -> int:
        try:
            return self._scan_codes[key]
        except KeyError as error:
            raise ValueError(f'不允许的按键: "{key}"') from error

    @staticmethod
    def _require_inserted(inserted_count: int) -> None:
        if inserted_count != 1:
            raise InputInjectionError(
                f"SendInput 应插入 1 个事件，实际插入 {inserted_count} 个"
            )

    def _check_new_action(self, *, now_ns: int) -> None:
        try:
            self._gate.check()
        except (EmergencyStopError, ForegroundWindowError) as error:
            self.release_all(now_ns=now_ns, reason=str(error))
            raise

    def key_down(self, key: str, *, now_ns: int) -> None:
        scan_code = self._scan_code(key)
        self._check_new_action(now_ns=now_ns)
        if key in self._pressed_at:
            return
        self._require_inserted(self._gateway.send_key(scan_code, key_up=False))
        self._pressed_at[key] = now_ns
        self._events.append(InputEvent("key_down", now_ns, key=key))

    def key_up(self, key: str, *, now_ns: int, reason: str | None = None) -> None:
        scan_code = self._scan_code(key)
        if key not in self._pressed_at:
            return
        # 释放动作不能再受前台窗口或急停闸门限制，否则可能留下卡键。
        self._require_inserted(self._gateway.send_key(scan_code, key_up=True))
        del self._pressed_at[key]
        self._events.append(InputEvent("key_up", now_ns, key=key, reason=reason))

    def move_mouse_relative(self, dx: int, dy: int, *, now_ns: int) -> None:
        self._check_new_action(now_ns=now_ns)
        self._require_inserted(self._gateway.send_mouse_relative(dx, dy))
        self._events.append(InputEvent("mouse_move", now_ns, dx=dx, dy=dy))

    def release_all(self, *, now_ns: int, reason: str) -> None:
        first_error: InputInjectionError | None = None
        # 使用按下顺序的逆序，优先释放后按下的组合键。
        for key in reversed(tuple(self._pressed_at)):
            try:
                self.key_up(key, now_ns=now_ns, reason=reason)
            except InputInjectionError as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def expire_overdue(self, *, now_ns: int) -> tuple[str, ...]:
        expired = tuple(
            sorted(
                key
                for key, pressed_at_ns in self._pressed_at.items()
                if now_ns - pressed_at_ns >= self._max_key_hold_ns
            )
        )
        first_error: InputInjectionError | None = None
        for key in expired:
            try:
                self.key_up(key, now_ns=now_ns, reason="超过最大按键时长")
            except InputInjectionError as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
        return expired
