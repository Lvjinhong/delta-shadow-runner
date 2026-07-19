import ctypes

import pytest

from delta_vision import win32_native
from delta_vision.win32_native import (
    IACE_DEFAULT,
    INPUT,
    INPUT_KEYBOARD,
    INPUT_MOUSE,
    KEYEVENTF_KEYUP,
    KEYEVENTF_SCANCODE,
    MOUSEEVENTF_ABSOLUTE,
    MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MOVE,
    MOUSEEVENTF_VIRTUALDESK,
    ImeDisabledSession,
    Win32NativeGateway,
    enable_per_monitor_dpi_awareness,
    find_window_handle,
    window_client_region,
    window_client_region_for_handle,
)


class FakeUser32:
    def __init__(self) -> None:
        self.title = "Delta Vision Test Target"
        self.input_desktop_name = "Default"
        self.key_state = 0
        self.inserted_count = 1
        self.inputs: list[INPUT] = []
        self.window_handle = 123
        self.input_desktop_handle = 456
        self.closed_desktop_handles: list[int] = []
        self.close_desktop_succeeds = True
        self.system_metrics = {
            76: -1920,
            77: 0,
            78: 4480,
            79: 1440,
        }

    def OpenInputDesktop(self, flags, inherit, desired_access):
        assert flags == 0
        assert inherit is False
        assert desired_access == 1
        return self.input_desktop_handle

    def GetUserObjectInformationW(
        self, desktop_handle, info_index, buffer, buffer_size, needed_pointer
    ):
        assert desktop_handle == self.input_desktop_handle
        assert info_index == 2
        encoded_size = (len(self.input_desktop_name) + 1) * ctypes.sizeof(
            ctypes.c_wchar
        )
        needed_pointer._obj.value = encoded_size
        if not buffer:
            assert buffer_size == 0
            return 0
        assert buffer_size == encoded_size
        source = ctypes.create_unicode_buffer(self.input_desktop_name)
        ctypes.memmove(buffer, source, encoded_size)
        return 1

    def CloseDesktop(self, desktop_handle):
        self.closed_desktop_handles.append(desktop_handle)
        return int(self.close_desktop_succeeds)

    def GetForegroundWindow(self):
        return self.window_handle

    def GetWindowTextLengthW(self, window_handle):
        assert window_handle == self.window_handle
        return len(self.title)

    def GetWindowTextW(self, window_handle, buffer, buffer_size):
        assert window_handle == self.window_handle
        assert buffer_size == len(self.title) + 1
        buffer.value = self.title
        return len(self.title)

    def GetAsyncKeyState(self, virtual_key):
        assert virtual_key == 0x7B
        return self.key_state

    def SendInput(self, count, events, event_size):
        assert count == 1
        assert event_size == ctypes.sizeof(INPUT)
        self.inputs.append(INPUT.from_buffer_copy(events[0]))
        return self.inserted_count

    def GetSystemMetrics(self, index):
        return self.system_metrics[index]

    def FindWindowW(self, class_name, window_title):
        assert class_name is None
        return self.window_handle if window_title == self.title else 0

    def GetClientRect(self, window_handle, rect_pointer):
        assert window_handle == self.window_handle
        rect = rect_pointer._obj
        rect.left = 0
        rect.top = 0
        rect.right = 1280
        rect.bottom = 720
        return 1

    def ClientToScreen(self, window_handle, point_pointer):
        assert window_handle == self.window_handle
        point = point_pointer._obj
        point.x += 100
        point.y += 200
        return 1


class FakeWtsApi32:
    def __init__(self, *, connection_state: int = 0) -> None:
        self.connection_state = ctypes.c_int(connection_state)
        self.freed_buffers: list[int] = []

    def WTSQuerySessionInformationW(
        self, server, session_id, info_class, buffer_pointer, bytes_pointer
    ):
        assert server in {0, None}
        assert session_id == 0xFFFFFFFF
        assert info_class == 8
        buffer_pointer._obj.value = ctypes.addressof(self.connection_state)
        bytes_pointer._obj.value = ctypes.sizeof(self.connection_state)
        return 1

    def WTSFreeMemory(self, buffer):
        self.freed_buffers.append(int(buffer.value))


class FakeImm32:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []
        self.failures: set[tuple[int, int]] = set()

    @staticmethod
    def _value(pointer) -> int:
        return int(getattr(pointer, "value", pointer) or 0)

    def ImmAssociateContextEx(self, window_handle, input_context, flags):
        handle = self._value(window_handle)
        context = self._value(input_context)
        effective_flags = self._value(flags)
        self.calls.append((handle, context, effective_flags))
        return int((handle, effective_flags) not in self.failures)


def test_gateway_reads_foreground_title_and_emergency_key() -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    assert gateway.foreground_title() == "Delta Vision Test Target"
    assert gateway.foreground_window_handle() == 123
    assert gateway.is_key_pressed(0x7B) is False
    user32.key_state = -32768
    assert gateway.is_key_pressed(0x7B) is True


def test_gateway_sends_scan_code_key_down_and_key_up() -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    assert gateway.send_key(0x11, key_up=False) == 1
    assert gateway.send_key(0x11, key_up=True) == 1

    down, up = user32.inputs
    assert down.type == INPUT_KEYBOARD
    assert down.ki.wVk == 0
    assert down.ki.wScan == 0x11
    assert down.ki.dwFlags == KEYEVENTF_SCANCODE
    assert up.ki.dwFlags == KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP


def test_gateway_sends_relative_mouse_motion() -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    assert gateway.send_mouse_relative(12, -7) == 1

    event = user32.inputs[0]
    assert event.type == INPUT_MOUSE
    assert event.mi.dx == 12
    assert event.mi.dy == -7
    assert event.mi.dwFlags == MOUSEEVENTF_MOVE


def test_gateway_sends_absolute_mouse_motion_on_virtual_desktop() -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    assert gateway.send_mouse_absolute(0, 720) == 1

    event = user32.inputs[0]
    assert event.type == INPUT_MOUSE
    assert event.mi.dx == round(1920 * 65_535 / 4479)
    assert event.mi.dy == round(720 * 65_535 / 1439)
    assert event.mi.dwFlags == (
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    )


@pytest.mark.parametrize(
    ("screen_x", "screen_y", "normalized_x", "normalized_y"),
    [
        (-1920, 0, 0, 0),
        (2559, 0, 65_535, 0),
        (-1920, 1439, 0, 65_535),
        (2559, 1439, 65_535, 65_535),
    ],
)
def test_gateway_maps_every_virtual_desktop_corner_exactly(
    screen_x: int,
    screen_y: int,
    normalized_x: int,
    normalized_y: int,
) -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    assert gateway.send_mouse_absolute(screen_x, screen_y) == 1

    event = user32.inputs[0]
    assert (event.mi.dx, event.mi.dy) == (normalized_x, normalized_y)


@pytest.mark.parametrize(("metric", "value"), [(78, 1), (79, 0)])
def test_gateway_rejects_invalid_virtual_desktop_size_without_input(
    metric: int,
    value: int,
) -> None:
    user32 = FakeUser32()
    user32.system_metrics[metric] = value
    gateway = Win32NativeGateway(user32=user32)

    with pytest.raises(ValueError, match="虚拟桌面尺寸"):
        gateway.send_mouse_absolute(0, 0)

    assert user32.inputs == []


@pytest.mark.parametrize(
    ("x", "y"),
    [(-1921, 0), (2560, 0), (0, -1), (0, 1440), (1.5, 20), (20, True)],
)
def test_gateway_rejects_absolute_mouse_position_outside_virtual_desktop(
    x: object,
    y: object,
) -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    with pytest.raises(ValueError, match="虚拟桌面"):
        gateway.send_mouse_absolute(x, y)  # type: ignore[arg-type]

    assert user32.inputs == []


def test_gateway_sends_left_mouse_down_and_up() -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    assert gateway.send_mouse_left(key_up=False) == 1
    assert gateway.send_mouse_left(key_up=True) == 1

    down, up = user32.inputs
    assert down.type == INPUT_MOUSE
    assert down.mi.dwFlags == MOUSEEVENTF_LEFTDOWN
    assert up.mi.dwFlags == MOUSEEVENTF_LEFTUP


@pytest.mark.parametrize("key_up", [0, 1, None, "false"])
def test_gateway_rejects_non_boolean_left_mouse_state_without_input(
    key_up: object,
) -> None:
    user32 = FakeUser32()
    gateway = Win32NativeGateway(user32=user32)

    with pytest.raises(ValueError, match="key_up"):
        gateway.send_mouse_left(key_up=key_up)  # type: ignore[arg-type]

    assert user32.inputs == []


def test_ime_disabled_session_uses_public_api_for_each_unique_window() -> None:
    imm32 = FakeImm32()
    session = ImeDisabledSession((101, 202, 101), imm32=imm32)

    session.disable()

    assert imm32.calls == [(101, 0, 0), (202, 0, 0)]


def test_load_imm32_uses_pointer_safe_associate_context_ex_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFunction:
        argtypes = None
        restype = None

    class FakeImm32Library:
        ImmAssociateContextEx = FakeFunction()

    library = FakeImm32Library()
    monkeypatch.setattr(win32_native.sys, "platform", "win32")
    monkeypatch.setattr(
        win32_native.ctypes,
        "WinDLL",
        lambda name, use_last_error: library,
        raising=False,
    )

    assert win32_native._load_imm32() is library
    assert library.ImmAssociateContextEx.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        win32_native.DWORD,
    ]
    assert library.ImmAssociateContextEx.restype is ctypes.c_int


def test_ime_disabled_session_rolls_back_when_disable_fails() -> None:
    imm32 = FakeImm32()
    imm32.failures.add((202, 0))
    session = ImeDisabledSession((101, 202, 303), imm32=imm32)

    with pytest.raises(OSError, match=r"禁用.*IME"):
        session.disable()

    assert imm32.calls == [
        (101, 0, 0),
        (202, 0, 0),
        (101, 0, IACE_DEFAULT),
    ]


def test_ime_disabled_session_restores_in_reverse_order_and_is_idempotent() -> None:
    imm32 = FakeImm32()
    session = ImeDisabledSession((101, 202), imm32=imm32)
    session.disable()

    session.restore()
    session.restore()

    assert imm32.calls == [
        (101, 0, 0),
        (202, 0, 0),
        (202, 0, IACE_DEFAULT),
        (101, 0, IACE_DEFAULT),
    ]


def test_ime_disabled_session_attempts_all_restores_after_one_failure() -> None:
    imm32 = FakeImm32()
    session = ImeDisabledSession((101, 202), imm32=imm32)
    session.disable()
    imm32.failures.add((202, IACE_DEFAULT))

    with pytest.raises(OSError, match=r"恢复.*IME"):
        session.restore()

    assert imm32.calls[-2:] == [
        (202, 0, IACE_DEFAULT),
        (101, 0, IACE_DEFAULT),
    ]
    imm32.failures.clear()
    session.restore()
    assert imm32.calls[-1] == (202, 0, IACE_DEFAULT)


def test_window_client_region_converts_client_origin_to_screen_coordinates() -> None:
    user32 = FakeUser32()
    wtsapi32 = FakeWtsApi32()

    region = window_client_region(
        "Delta Vision Test Target",
        user32=user32,
        wtsapi32=wtsapi32,
    )

    assert (region.left, region.top, region.width, region.height) == (100, 200, 1280, 720)
    assert user32.closed_desktop_handles == [456]


def test_window_client_region_for_handle_uses_exact_bound_window() -> None:
    user32 = FakeUser32()

    region = window_client_region_for_handle(
        123,
        user32=user32,
        wtsapi32=FakeWtsApi32(),
    )

    assert (region.left, region.top, region.width, region.height) == (
        100,
        200,
        1280,
        720,
    )


def test_window_client_region_rejects_locked_input_desktop_before_capture() -> None:
    user32 = FakeUser32()
    user32.input_desktop_name = "Winlogon"

    with pytest.raises(RuntimeError, match=r"锁屏.*Winlogon"):
        window_client_region(
            "Delta Vision Test Target",
            user32=user32,
            wtsapi32=FakeWtsApi32(),
        )

    assert user32.closed_desktop_handles == [456]


def test_window_client_region_rejects_disconnected_default_desktop() -> None:
    user32 = FakeUser32()
    wtsapi32 = FakeWtsApi32(connection_state=4)

    with pytest.raises(RuntimeError, match=r"会话.*Disconnected"):
        window_client_region(
            "Delta Vision Test Target",
            user32=user32,
            wtsapi32=wtsapi32,
        )

    assert wtsapi32.freed_buffers == [ctypes.addressof(wtsapi32.connection_state)]


def test_window_client_region_rejects_failed_desktop_handle_cleanup() -> None:
    user32 = FakeUser32()
    user32.close_desktop_succeeds = False

    with pytest.raises(OSError, match="关闭 Windows 输入桌面句柄失败"):
        window_client_region(
            "Delta Vision Test Target",
            user32=user32,
            wtsapi32=FakeWtsApi32(),
        )


def test_find_window_handle_resolves_exact_title() -> None:
    user32 = FakeUser32()

    assert find_window_handle("Delta Vision Test Target", user32=user32) == 123


def test_enable_per_monitor_dpi_awareness_uses_v2_context() -> None:
    class FakeDpiUser32:
        def __init__(self) -> None:
            self.context = None

        def SetProcessDpiAwarenessContext(self, context):
            self.context = context.value
            return 1

    user32 = FakeDpiUser32()

    enable_per_monitor_dpi_awareness(user32=user32)

    assert user32.context is not None
    assert user32.context != 0


def test_window_client_region_rejects_missing_window() -> None:
    with pytest.raises(LookupError, match="找不到窗口"):
        window_client_region(
            "Missing Window",
            user32=FakeUser32(),
            wtsapi32=FakeWtsApi32(),
        )
