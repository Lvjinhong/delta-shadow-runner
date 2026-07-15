"""Windows 屏幕采集适配器。

模块导入时不加载 Windows 专属依赖，便于在其他平台运行离线回放和单元测试。
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import CaptureRegion
from .frames import CapturedFrame


class _DxcamCamera(Protocol):
    def grab(self, **kwargs: object) -> object: ...

    def release(self) -> None: ...


class _MssSession(Protocol):
    def grab(self, region: dict[str, int]) -> object: ...

    def close(self) -> None: ...


def _owned_bgr_image(frame: object) -> NDArray[np.uint8]:
    """把 BGRA 帧转成独立、只读的 BGR 图像。"""

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"截图必须是 H×W×4 的 BGRA 图像，实际形状: {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"BGRA 图像必须使用 uint8，实际类型: {image.dtype}")
    bgr = np.array(image[:, :, :3], dtype=np.uint8, copy=True, order="C")
    bgr.setflags(write=False)
    return bgr


class DxcamFrameSource:
    """使用 DXGI Desktop Duplication 获取指定屏幕区域。"""

    def __init__(
        self,
        region: CaptureRegion,
        *,
        device_idx: int = 0,
        output_idx: int = 0,
        dxcam_module: ModuleType | Any | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        module = dxcam_module or importlib.import_module("dxcam")
        self._camera: _DxcamCamera = module.create(
            device_idx=device_idx,
            output_idx=output_idx,
            output_color="BGRA",
            backend="dxgi",
            processor_backend="numpy",
        )
        self._region = region
        self._source = f"dxcam:{device_idx}:{output_idx}"
        self._clock_ns = clock_ns
        self._sequence = 0
        self._closed = False
        self._last_capture_duration_ns: int | None = None

    @property
    def last_capture_duration_ns(self) -> int | None:
        return self._last_capture_duration_ns

    def grab(self) -> CapturedFrame | None:
        if self._closed:
            raise RuntimeError("DXcam 截图源已经关闭")
        started_at_ns = self._clock_ns()
        raw_frame = self._camera.grab(
            region=self._region.as_dxcam(),
            copy=True,
            new_frame_only=False,
        )
        captured_at_ns = self._clock_ns()
        self._last_capture_duration_ns = max(0, captured_at_ns - started_at_ns)
        if raw_frame is None:
            return None
        image = _owned_bgr_image(raw_frame)
        result = CapturedFrame(
            sequence=self._sequence,
            captured_at_ns=captured_at_ns,
            image=image,
            source=self._source,
        )
        self._sequence += 1
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._camera.release()
        self._closed = True

    def __enter__(self) -> DxcamFrameSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class MssFrameSource:
    """使用 MSS/GDI 获取指定屏幕区域，作为 DXcam 的兼容回退。"""

    def __init__(
        self,
        region: CaptureRegion,
        *,
        mss_factory: Callable[[], _MssSession] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if mss_factory is None:
            module = importlib.import_module("mss")
            mss_factory = module.mss
        self._session = mss_factory()
        self._region = region
        self._clock_ns = clock_ns
        self._sequence = 0
        self._closed = False

    def grab(self) -> CapturedFrame:
        if self._closed:
            raise RuntimeError("MSS 截图源已经关闭")
        raw_frame = self._session.grab(
            {
                "left": self._region.left,
                "top": self._region.top,
                "width": self._region.width,
                "height": self._region.height,
            }
        )
        image = _owned_bgr_image(raw_frame)
        result = CapturedFrame(
            sequence=self._sequence,
            captured_at_ns=self._clock_ns(),
            image=image,
            source="mss",
        )
        self._sequence += 1
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._session.close()
        self._closed = True

    def __enter__(self) -> MssFrameSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
