import ctypes

import pytest

from delta_vision.win32_native import (
    INPUT,
    INPUT_KEYBOARD,
    INPUT_MOUSE,
    KEYEVENTF_KEYUP,
    KEYEVENTF_SCANCODE,
    MOUSEEVENTF_MOVE,
    Win32NativeGateway,
    enable_per_monitor_dpi_awareness,
    find_window_handle,
    window_client_region,
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
