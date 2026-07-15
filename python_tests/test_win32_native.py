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
    window_client_region,
)


class FakeUser32:
    def __init__(self) -> None:
        self.title = "Delta Vision Test Target"
        self.key_state = 0
        self.inserted_count = 1
        self.inputs: list[INPUT] = []
        self.window_handle = 123

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
    region = window_client_region("Delta Vision Test Target", user32=FakeUser32())

    assert (region.left, region.top, region.width, region.height) == (100, 200, 1280, 720)


def test_window_client_region_rejects_missing_window() -> None:
    with pytest.raises(LookupError, match="找不到窗口"):
        window_client_region("Missing Window", user32=FakeUser32())
