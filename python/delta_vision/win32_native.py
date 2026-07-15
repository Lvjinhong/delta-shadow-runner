"""对 Windows user32 的最小标准输入和窗口查询封装。"""

from __future__ import annotations

import ctypes
import sys
from ctypes import Structure, Union, c_int32, c_uint16, c_uint32, c_uint64
from typing import Any, ClassVar

from .config import CaptureRegion

WORD = c_uint16
DWORD = c_uint32
LONG = c_int32
ULONG_PTR = c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else c_uint32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


class MOUSEINPUT(Structure):
    _fields_ = [
        ("dx", LONG),
        ("dy", LONG),
        ("mouseData", DWORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(Structure):
    _fields_ = [
        ("wVk", WORD),
        ("wScan", WORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(Union):
    _fields_: ClassVar[list[tuple[str, type[Structure]]]] = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
    ]


class INPUT(Structure):
    _anonymous_ = ("payload",)
    _fields_ = [("type", DWORD), ("payload", _INPUTUNION)]


class POINT(Structure):
    _fields_ = [("x", LONG), ("y", LONG)]


class RECT(Structure):
    _fields_ = [("left", LONG), ("top", LONG), ("right", LONG), ("bottom", LONG)]


def _load_user32() -> Any:
    if sys.platform != "win32":
        raise OSError("Win32 原生输入只能在 Windows 上运行")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    # ctypes 默认把返回值当 c_int；显式签名避免 64 位 HWND 被截断。
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = ctypes.c_uint
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(RECT)]
    user32.GetClientRect.restype = ctypes.c_int
    user32.ClientToScreen.argtypes = [ctypes.c_void_p, ctypes.POINTER(POINT)]
    user32.ClientToScreen.restype = ctypes.c_int
    if hasattr(user32, "SetProcessDpiAwarenessContext"):
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_int
    if hasattr(user32, "SetProcessDPIAware"):
        user32.SetProcessDPIAware.argtypes = []
        user32.SetProcessDPIAware.restype = ctypes.c_int
    return user32


def _last_error() -> int:
    getter = getattr(ctypes, "get_last_error", None)
    return 0 if getter is None else int(getter())


def enable_per_monitor_dpi_awareness(*, user32: Any | None = None) -> None:
    """让窗口客户区坐标与 DXGI 物理像素使用同一坐标系。"""

    native = user32 or _load_user32()
    setter = getattr(native, "SetProcessDpiAwarenessContext", None)
    if setter is not None and setter(ctypes.c_void_p(-4)):
        return
    fallback = getattr(native, "SetProcessDPIAware", None)
    if fallback is not None and fallback():
        return
    raise OSError(_last_error(), "无法启用进程 DPI Awareness")


class Win32NativeGateway:
    """只使用公开 user32 API，不安装 hook，也不读取其他进程。"""

    def __init__(self, *, user32: Any | None = None) -> None:
        self._user32 = user32 or _load_user32()

    def foreground_window_handle(self) -> int:
        return int(self._user32.GetForegroundWindow() or 0)

    def foreground_title(self) -> str:
        window_handle = self.foreground_window_handle()
        if not window_handle:
            return ""
        title_length = int(self._user32.GetWindowTextLengthW(window_handle))
        buffer = ctypes.create_unicode_buffer(title_length + 1)
        copied = int(
            self._user32.GetWindowTextW(window_handle, buffer, title_length + 1)
        )
        if copied <= 0:
            return ""
        return buffer.value

    def is_key_pressed(self, virtual_key: int) -> bool:
        return bool(int(self._user32.GetAsyncKeyState(virtual_key)) & 0x8000)

    def send_key(self, scan_code: int, *, key_up: bool) -> int:
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else 0)
        event = INPUT(
            type=INPUT_KEYBOARD,
            payload=_INPUTUNION(
                ki=KEYBDINPUT(
                    wVk=0,
                    wScan=scan_code,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )
        events = (INPUT * 1)(event)
        return int(self._user32.SendInput(1, events, ctypes.sizeof(INPUT)))

    def send_mouse_relative(self, dx: int, dy: int) -> int:
        event = INPUT(
            type=INPUT_MOUSE,
            payload=_INPUTUNION(
                mi=MOUSEINPUT(
                    dx=dx,
                    dy=dy,
                    mouseData=0,
                    dwFlags=MOUSEEVENTF_MOVE,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )
        events = (INPUT * 1)(event)
        return int(self._user32.SendInput(1, events, ctypes.sizeof(INPUT)))


def window_client_region(
    window_title: str,
    *,
    user32: Any | None = None,
) -> CaptureRegion:
    """把指定顶层窗口的客户区转换成桌面像素坐标。"""

    native = user32 or _load_user32()
    if user32 is None:
        enable_per_monitor_dpi_awareness(user32=native)
    window_handle = find_window_handle(window_title, user32=native)

    rect = RECT()
    if not native.GetClientRect(window_handle, ctypes.byref(rect)):
        raise OSError(_last_error(), "GetClientRect 失败")
    origin = POINT(rect.left, rect.top)
    if not native.ClientToScreen(window_handle, ctypes.byref(origin)):
        raise OSError(_last_error(), "ClientToScreen 失败")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    return CaptureRegion(int(origin.x), int(origin.y), width, height)


def find_window_handle(window_title: str, *, user32: Any | None = None) -> int:
    """按完整标题解析顶层窗口句柄，用于启动时绑定安全门。"""

    if not window_title:
        raise ValueError("窗口标题不能为空")
    native = user32 or _load_user32()
    window_handle = int(native.FindWindowW(None, window_title) or 0)
    if not window_handle:
        raise LookupError(f'找不到窗口: "{window_title}"')
    return window_handle
