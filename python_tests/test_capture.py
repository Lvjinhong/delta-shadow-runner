import numpy as np
import pytest

from delta_vision.capture import DxcamFrameSource, MssFrameSource
from delta_vision.config import CaptureRegion


class FakeDxcamCamera:
    def __init__(self, frames) -> None:
        self.frames = iter(frames)
        self.grab_calls = []
        self.released = False

    def grab(self, **kwargs):
        self.grab_calls.append(kwargs)
        return next(self.frames)

    def release(self) -> None:
        self.released = True


class FakeDxcamModule:
    def __init__(self, camera: FakeDxcamCamera) -> None:
        self.camera = camera
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.camera


def test_dxcam_source_converts_bgra_to_owned_read_only_bgr_frame() -> None:
    bgra = np.zeros((4, 6, 4), dtype=np.uint8)
    bgra[:, :, :3] = (10, 20, 30)
    bgra[:, :, 3] = 255
    camera = FakeDxcamCamera([bgra])
    module = FakeDxcamModule(camera)
    clock = iter([100, 200])
    source = DxcamFrameSource(
        CaptureRegion(5, 7, 6, 4),
        device_idx=1,
        output_idx=2,
        dxcam_module=module,
        clock_ns=lambda: next(clock),
    )

    frame = source.grab()

    assert frame is not None
    assert frame.sequence == 0
    assert frame.captured_at_ns == 200
    assert frame.source == "dxcam:1:2"
    assert frame.image.shape == (4, 6, 3)
    assert tuple(frame.image[0, 0]) == (10, 20, 30)
    assert frame.image.flags.writeable is False
    bgra[:, :, :3] = 99
    assert tuple(frame.image[0, 0]) == (10, 20, 30)
    assert module.create_calls == [
        {
            "device_idx": 1,
            "output_idx": 2,
            "output_color": "BGRA",
            "backend": "dxgi",
            "processor_backend": "numpy",
        }
    ]
    assert camera.grab_calls == [
        {"region": (5, 7, 11, 11), "copy": True, "new_frame_only": False}
    ]


def test_dxcam_source_does_not_advance_sequence_when_no_frame() -> None:
    image = np.zeros((2, 3, 4), dtype=np.uint8)
    camera = FakeDxcamCamera([None, image])
    source = DxcamFrameSource(
        CaptureRegion(0, 0, 3, 2),
        dxcam_module=FakeDxcamModule(camera),
        clock_ns=lambda: 1,
    )

    assert source.grab() is None
    assert source.grab().sequence == 0


def test_dxcam_source_releases_camera() -> None:
    camera = FakeDxcamCamera([None])
    source = DxcamFrameSource(
        CaptureRegion(0, 0, 2, 2),
        dxcam_module=FakeDxcamModule(camera),
    )

    source.close()
    source.close()

    assert camera.released is True


def test_dxcam_source_allows_release_retry_after_failure() -> None:
    class FlakyCamera(FakeDxcamCamera):
        def __init__(self) -> None:
            super().__init__([None])
            self.release_calls = 0

        def release(self) -> None:
            self.release_calls += 1
            if self.release_calls == 1:
                raise RuntimeError("temporary release failure")
            self.released = True

    camera = FlakyCamera()
    source = DxcamFrameSource(
        CaptureRegion(0, 0, 2, 2),
        dxcam_module=FakeDxcamModule(camera),
    )

    with pytest.raises(RuntimeError, match="temporary"):
        source.close()
    source.close()

    assert camera.release_calls == 2
    assert camera.released is True


def test_dxcam_source_rejects_unexpected_frame_shape() -> None:
    camera = FakeDxcamCamera([np.zeros((3, 4), dtype=np.uint8)])
    source = DxcamFrameSource(
        CaptureRegion(0, 0, 4, 3),
        dxcam_module=FakeDxcamModule(camera),
    )

    with pytest.raises(ValueError, match="BGRA"):
        source.grab()


class FakeMssSession:
    def __init__(self, frame) -> None:
        self.frame = frame
        self.calls = []
        self.closed = False

    def grab(self, region):
        self.calls.append(region)
        return self.frame

    def close(self) -> None:
        self.closed = True


def test_mss_source_uses_xywh_region_and_bgra_conversion() -> None:
    bgra = np.full((3, 4, 4), 7, dtype=np.uint8)
    session = FakeMssSession(bgra)
    source = MssFrameSource(
        CaptureRegion(-100, 20, 4, 3),
        mss_factory=lambda: session,
        clock_ns=lambda: 123,
    )

    frame = source.grab()
    source.close()

    assert frame.sequence == 0
    assert frame.captured_at_ns == 123
    assert frame.source == "mss"
    assert frame.image.shape == (3, 4, 3)
    assert session.calls == [{"left": -100, "top": 20, "width": 4, "height": 3}]
    assert session.closed is True
