"""带前台窗口和急停保护的标准 Win32 输入执行器。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
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


class TimerHandle(Protocol):
    def start(self) -> None: ...

    def cancel(self) -> None: ...


def _daemon_timer(interval_seconds: float, callback: Callable[[], None]) -> TimerHandle:
    timer = threading.Timer(interval_seconds, callback)
    timer.daemon = True
    return timer


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

    _WATCHDOG_RETRY_SECONDS = 0.01

    def __init__(
        self,
        *,
        scan_codes: dict[str, int],
        max_key_hold_ms: int,
        gate: SafetyGate,
        gateway: InputGateway,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        timer_factory: Callable[[float, Callable[[], None]], TimerHandle] = _daemon_timer,
    ) -> None:
        if not scan_codes:
            raise ValueError("scan code 映射不能为空")
        if isinstance(max_key_hold_ms, bool) or max_key_hold_ms <= 0:
            raise ValueError("最大按键时长必须为正数")
        self._scan_codes = dict(scan_codes)
        self._max_key_hold_ns = max_key_hold_ms * 1_000_000
        self._max_key_hold_seconds = max_key_hold_ms / 1_000
        self._gate = gate
        self._gateway = gateway
        self._clock_ns = clock_ns
        self._timer_factory = timer_factory
        self._state_lock = threading.RLock()
        self._key_locks = {key: threading.RLock() for key in self._scan_codes}
        self._pressed_at: dict[str, tuple[int, object]] = {}
        self._watchdog_timers: dict[str, tuple[object, TimerHandle]] = {}
        self._watchdog_error: InputInjectionError | None = None
        self._events: list[InputEvent] = []
        self._last_event_at_ns = -1

    @property
    def pressed_keys(self) -> frozenset[str]:
        with self._state_lock:
            return frozenset(self._pressed_at)

    @property
    def events(self) -> tuple[InputEvent, ...]:
        with self._state_lock:
            return tuple(self._events)

    def _scan_code(self, key: str) -> int:
        try:
            return self._scan_codes[key]
        except KeyError as error:
            raise ValueError(f'不允许的按键: "{key}"') from error

    def _next_event_time_locked(self, now_ns: int) -> int:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("动作时间戳必须是非负整数")
        effective_now_ns = max(now_ns, self._last_event_at_ns)
        self._last_event_at_ns = effective_now_ns
        return effective_now_ns

    @staticmethod
    def _require_inserted(inserted_count: int) -> None:
        if inserted_count != 1:
            raise InputInjectionError(
                f"SendInput 应插入 1 个事件，实际插入 {inserted_count} 个"
            )

    def _check_new_action(self, *, now_ns: int) -> None:
        with self._state_lock:
            watchdog_error = self._watchdog_error
            self._watchdog_error = None
        if watchdog_error is not None:
            try:
                self.release_all(now_ns=now_ns, reason="看门狗释放按键失败")
            except InputInjectionError as release_error:
                self._store_watchdog_error(release_error)
            raise watchdog_error
        try:
            self._gate.check()
        except (EmergencyStopError, ForegroundWindowError) as error:
            self.release_all(now_ns=now_ns, reason=str(error))
            raise

    def _store_watchdog_error(self, error: InputInjectionError) -> None:
        with self._state_lock:
            self._watchdog_error = self._watchdog_error or error

    def _schedule_watchdog_retry(self, key: str, token: object) -> None:
        timer = self._timer_factory(
            self._WATCHDOG_RETRY_SECONDS,
            lambda: self._watchdog_release(key, token),
        )
        with self._state_lock:
            current = self._pressed_at.get(key)
            if current is None or current[1] is not token:
                return
            self._watchdog_timers[key] = (token, timer)
        try:
            timer.start()
        except Exception as error:
            self._store_watchdog_error(
                InputInjectionError("无法启动按键释放看门狗重试")
            )
            raise InputInjectionError("无法启动按键释放看门狗重试") from error

    def _watchdog_release(self, key: str, token: object) -> None:
        try:
            self._release_key(
                key,
                now_ns=self._clock_ns(),
                reason="看门狗超过最大按键时长",
                expected_token=token,
            )
        except InputInjectionError as error:
            # 控制线程即使卡在抓帧，后台线程也会继续重试释放同一轮按键。
            self._store_watchdog_error(error)
            try:
                self._schedule_watchdog_retry(key, token)
            except InputInjectionError as retry_error:
                self._store_watchdog_error(retry_error)

    def key_down(self, key: str, *, now_ns: int) -> None:
        scan_code = self._scan_code(key)
        self._check_new_action(now_ns=now_ns)
        with self._key_locks[key]:
            with self._state_lock:
                if key in self._pressed_at:
                    return
            token = object()
            timer = self._timer_factory(
                self._max_key_hold_seconds,
                lambda: self._watchdog_release(key, token),
            )
            self._require_inserted(self._gateway.send_key(scan_code, key_up=False))
            with self._state_lock:
                effective_now_ns = self._next_event_time_locked(now_ns)
                self._pressed_at[key] = (effective_now_ns, token)
                self._watchdog_timers[key] = (token, timer)
                self._events.append(InputEvent("key_down", effective_now_ns, key=key))
            try:
                timer.start()
            except Exception as error:
                try:
                    self._release_key(
                        key,
                        now_ns=self._clock_ns(),
                        reason="看门狗启动失败，立即释放按键",
                        expected_token=token,
                    )
                except InputInjectionError as release_error:
                    self._store_watchdog_error(release_error)
                    raise InputInjectionError(
                        "无法启动按键释放看门狗，且立即释放也失败"
                    ) from release_error
                raise InputInjectionError("无法启动按键释放看门狗") from error

    def key_up(self, key: str, *, now_ns: int, reason: str | None = None) -> None:
        self._release_key(key, now_ns=now_ns, reason=reason)

    def _release_key(
        self,
        key: str,
        *,
        now_ns: int,
        reason: str | None,
        expected_token: object | None = None,
    ) -> bool:
        scan_code = self._scan_code(key)
        with self._key_locks[key]:
            with self._state_lock:
                current = self._pressed_at.get(key)
                if current is None:
                    return False
                token = current[1]
                if expected_token is not None and token is not expected_token:
                    return False
            # 释放动作不能再受前台窗口或急停闸门限制，否则可能留下卡键。
            self._require_inserted(self._gateway.send_key(scan_code, key_up=True))
            with self._state_lock:
                current = self._pressed_at.get(key)
                if current is None or current[1] is not token:
                    return False
                del self._pressed_at[key]
                timer_entry = self._watchdog_timers.get(key)
                timer = None
                if timer_entry is not None and timer_entry[0] is token:
                    del self._watchdog_timers[key]
                    timer = timer_entry[1]
                effective_now_ns = self._next_event_time_locked(now_ns)
                self._events.append(
                    InputEvent("key_up", effective_now_ns, key=key, reason=reason)
                )
            if timer is not None:
                timer.cancel()
            return True

    def move_mouse_relative(self, dx: int, dy: int, *, now_ns: int) -> None:
        self._check_new_action(now_ns=now_ns)
        self._require_inserted(self._gateway.send_mouse_relative(dx, dy))
        with self._state_lock:
            effective_now_ns = self._next_event_time_locked(now_ns)
            self._events.append(
                InputEvent("mouse_move", effective_now_ns, dx=dx, dy=dy)
            )

    def release_all(self, *, now_ns: int, reason: str) -> None:
        with self._state_lock:
            pressed = tuple(
                (key, state[1]) for key, state in reversed(self._pressed_at.items())
            )
        first_error: InputInjectionError | None = None
        for key, token in pressed:
            try:
                self._release_key(
                    key,
                    now_ns=now_ns,
                    reason=reason,
                    expected_token=token,
                )
            except InputInjectionError as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def expire_overdue(self, *, now_ns: int) -> tuple[str, ...]:
        with self._state_lock:
            watchdog_error = self._watchdog_error
            self._watchdog_error = None
            expired_states = tuple(
                sorted(
                    (key, state[1])
                    for key, state in self._pressed_at.items()
                    if now_ns - state[0] >= self._max_key_hold_ns
                )
            )
        first_error: InputInjectionError | None = None
        for key, token in expired_states:
            try:
                self._release_key(
                    key,
                    now_ns=now_ns,
                    reason="超过最大按键时长",
                    expected_token=token,
                )
            except InputInjectionError as error:
                first_error = first_error or error
        if first_error is not None:
            self._store_watchdog_error(first_error)
            raise first_error
        if watchdog_error is not None:
            raise watchdog_error
        return tuple(key for key, _ in expired_states)
