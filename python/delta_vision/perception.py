"""受控测试窗口使用的可解释颜色锚点感知。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AnchorObservation:
    label: str
    centroid: tuple[float, float] | None
    candidate_centroid: tuple[float, float] | None
    area: int
    confidence: float
    accepted: bool


class ColorAnchorDetector:
    """检测一个已知 BGR 色块，用于证明黑盒截图闭环。"""

    def __init__(
        self,
        *,
        label: str,
        bgr: tuple[int, int, int],
        tolerance: int,
        minimum_area: int,
        confidence_threshold: float,
    ) -> None:
        if not label:
            raise ValueError("锚点标签不能为空")
        if tolerance < 0:
            raise ValueError("颜色容差不能为负数")
        if minimum_area <= 0:
            raise ValueError("最小面积必须为正数")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("置信度阈值必须大于 0 且不超过 1")
        if any(channel < 0 or channel > 255 for channel in bgr):
            raise ValueError("BGR 通道必须位于 0 到 255 之间")

        self._label = label
        self._bgr = np.asarray(bgr, dtype=np.int16)
        self._tolerance = tolerance
        self._minimum_area = minimum_area
        self._confidence_threshold = confidence_threshold

    def detect(self, image: NDArray[np.uint8]) -> AnchorObservation:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("锚点检测要求 H×W×3 的 uint8 BGR 图像")

        pixels = image[:, :, :3].astype(np.int16, copy=False)
        mask = np.all(np.abs(pixels - self._bgr) <= self._tolerance, axis=2)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        component_index = None
        if count > 1:
            component_index = max(
                range(1, count),
                key=lambda index: (int(stats[index, cv2.CC_STAT_AREA]), -index),
            )
        area = 0 if component_index is None else int(stats[component_index, cv2.CC_STAT_AREA])
        confidence = min(1.0, area / self._minimum_area)
        candidate_centroid = None
        if component_index is not None:
            candidate_centroid = (
                float(centroids[component_index][0]),
                float(centroids[component_index][1]),
            )
        accepted = confidence >= self._confidence_threshold

        return AnchorObservation(
            label=self._label,
            centroid=candidate_centroid if accepted else None,
            candidate_centroid=candidate_centroid,
            area=area,
            confidence=confidence,
            accepted=accepted,
        )
