"""固定 ROI 内的可解释多尺度模板锚点匹配。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import CaptureRegion
from .frames import CapturedFrame
from .navigation import WaypointObservation

MatchReason = Literal["accepted", "below_threshold", "ambiguous", "no_candidate"]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class MatchDecisionPolicy:
    """把匹配分数和次优差距转换为可执行或拒绝决策。"""

    score_threshold: float
    minimum_margin: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.score_threshold)
            or not 0 < self.score_threshold <= 1
        ):
            raise ValueError("匹配分数阈值必须大于 0 且不超过 1")
        if (
            not math.isfinite(self.minimum_margin)
            or not 0 <= self.minimum_margin <= 1
        ):
            raise ValueError("最佳与次佳分数差必须位于 0 到 1 之间")

    def decide(self, *, best_score: float, runner_up_score: float) -> MatchReason:
        if not math.isfinite(best_score) or not math.isfinite(runner_up_score):
            raise ValueError("模板匹配分数必须是有限数")
        if best_score < self.score_threshold:
            return "below_threshold"
        if best_score - runner_up_score < self.minimum_margin:
            return "ambiguous"
        return "accepted"


@dataclass(frozen=True, slots=True)
class TemplateMatchObservation:
    label: str
    confidence: float
    runner_up_confidence: float
    bbox: BBox | None
    centroid: tuple[float, float] | None
    scale: float | None
    candidate_bbox: BBox | None
    candidate_centroid: tuple[float, float] | None
    candidate_scale: float | None
    accepted: bool
    reason: MatchReason


@dataclass(frozen=True, slots=True)
class _Candidate:
    score: float
    bbox: BBox
    centroid: tuple[float, float]
    scale: float


class TemplateAnchorDetector:
    """在客户区截图的固定 ROI 内搜索一个视觉锚点。"""

    _CANDIDATES_PER_SCALE = 8

    def __init__(
        self,
        *,
        label: str,
        template: NDArray[np.uint8],
        search_roi: CaptureRegion,
        scales: tuple[float, ...],
        policy: MatchDecisionPolicy,
        nms_radius_px: int,
    ) -> None:
        if not label:
            raise ValueError("模板标签不能为空")
        self._validate_image(template, field="模板")
        if search_roi.left < 0 or search_roi.top < 0:
            raise ValueError("搜索 ROI 左上角不能为负数")
        if not scales:
            raise ValueError("至少需要一个模板缩放比例")
        if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError("模板缩放比例必须是正有限数")
        if isinstance(nms_radius_px, bool) or nms_radius_px <= 0:
            raise ValueError("NMS 半径必须为正整数")

        owned_template = np.array(template[:, :, :3], dtype=np.uint8, copy=True)
        gray_template = cv2.cvtColor(owned_template, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray_template)) < 1:
            raise ValueError("模板纹理不足，不能进行归一化相关匹配")
        owned_template.setflags(write=False)
        gray_template.setflags(write=False)

        self._label = label
        self._template = owned_template
        self._gray_template = gray_template
        self._search_roi = search_roi
        self._scales = tuple(scales)
        self._policy = policy
        self._nms_radius_px = nms_radius_px

    @staticmethod
    def _validate_image(image: NDArray[np.uint8], *, field: str) -> None:
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.shape[2] < 3
        ):
            raise ValueError(f"{field}必须是 H×W×3 的 uint8 BGR 图像")

    @staticmethod
    def _centroid(bbox: BBox) -> tuple[float, float]:
        x, y, width, height = bbox
        return (x + width / 2, y + height / 2)

    def _scaled_template(self, scale: float) -> NDArray[np.uint8]:
        height, width = self._gray_template.shape
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(
            self._gray_template,
            (target_width, target_height),
            interpolation=interpolation,
        )

    def _scale_candidates(
        self,
        search_gray: NDArray[np.uint8],
        *,
        scale: float,
    ) -> tuple[_Candidate, ...]:
        template = self._scaled_template(scale)
        template_height, template_width = template.shape
        search_height, search_width = search_gray.shape
        if template_width > search_width or template_height > search_height:
            return ()

        scores = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
        scores = np.nan_to_num(scores, copy=True, nan=-1, posinf=-1, neginf=-1)
        candidates: list[_Candidate] = []
        for _ in range(self._CANDIDATES_PER_SCALE):
            _, best_score, _, best_location = cv2.minMaxLoc(scores)
            if best_score < -0.5:
                break
            local_x, local_y = best_location
            bbox = (
                self._search_roi.left + local_x,
                self._search_roi.top + local_y,
                template_width,
                template_height,
            )
            candidates.append(
                _Candidate(
                    score=max(0.0, min(1.0, float(best_score))),
                    bbox=bbox,
                    centroid=self._centroid(bbox),
                    scale=scale,
                )
            )
            left = max(0, local_x - self._nms_radius_px)
            top = max(0, local_y - self._nms_radius_px)
            right = min(scores.shape[1], local_x + self._nms_radius_px + 1)
            bottom = min(scores.shape[0], local_y + self._nms_radius_px + 1)
            scores[top:bottom, left:right] = -1
        return tuple(candidates)

    def detect(self, image: NDArray[np.uint8]) -> TemplateMatchObservation:
        self._validate_image(image, field="待检测截图")
        if self._search_roi.right > image.shape[1] or self._search_roi.bottom > image.shape[0]:
            raise ValueError(
                "搜索 ROI 超出截图范围: "
                f"roi={self._search_roi.width}x{self._search_roi.height}"
                f"@({self._search_roi.left},{self._search_roi.top}), "
                f"frame={image.shape[1]}x{image.shape[0]}"
            )
        search_bgr = image[
            self._search_roi.top : self._search_roi.bottom,
            self._search_roi.left : self._search_roi.right,
            :3,
        ]
        search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
        candidates = sorted(
            (
                candidate
                for scale in self._scales
                for candidate in self._scale_candidates(search_gray, scale=scale)
            ),
            key=lambda candidate: (
                -candidate.score,
                candidate.bbox[1],
                candidate.bbox[0],
                candidate.scale,
            ),
        )
        if not candidates:
            return TemplateMatchObservation(
                label=self._label,
                confidence=0,
                runner_up_confidence=0,
                bbox=None,
                centroid=None,
                scale=None,
                candidate_bbox=None,
                candidate_centroid=None,
                candidate_scale=None,
                accepted=False,
                reason="no_candidate",
            )

        best = candidates[0]
        runner_up = next(
            (
                candidate
                for candidate in candidates[1:]
                if math.dist(candidate.centroid, best.centroid) > self._nms_radius_px
            ),
            None,
        )
        runner_up_score = 0 if runner_up is None else runner_up.score
        reason = self._policy.decide(
            best_score=best.score,
            runner_up_score=runner_up_score,
        )
        accepted = reason == "accepted"
        return TemplateMatchObservation(
            label=self._label,
            confidence=best.score,
            runner_up_confidence=runner_up_score,
            bbox=best.bbox if accepted else None,
            centroid=best.centroid if accepted else None,
            scale=best.scale if accepted else None,
            candidate_bbox=best.bbox,
            candidate_centroid=best.centroid,
            candidate_scale=best.scale,
            accepted=accepted,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class RouteTemplate:
    """把一个截图模板映射到路线画布坐标和可选 waypoint。"""

    template_id: str
    detector: TemplateAnchorDetector
    route_position: tuple[float, float]
    waypoint_id: str | None

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("路线模板 ID 不能为空")
        if len(self.route_position) != 2 or not all(
            math.isfinite(value) for value in self.route_position
        ):
            raise ValueError("路线坐标必须包含两个有限数")
        if self.waypoint_id is not None and not self.waypoint_id:
            raise ValueError("waypoint ID 必须是非空字符串或 null")


class TemplateWaypointObserver:
    """从多个关键帧模板中选择唯一路线位置，拒绝跨模板歧义。"""

    def __init__(
        self,
        *,
        templates: tuple[RouteTemplate, ...],
        expected_frame_size: tuple[int, int],
        minimum_template_margin: float,
    ) -> None:
        if not templates:
            raise ValueError("路线模板不能为空")
        template_ids = tuple(template.template_id for template in templates)
        if len(set(template_ids)) != len(template_ids):
            raise ValueError("路线模板 ID 不能重复")
        if (
            len(expected_frame_size) != 2
            or any(type(value) is not int or value <= 0 for value in expected_frame_size)
        ):
            raise ValueError("期望帧分辨率必须包含两个正整数")
        if (
            not math.isfinite(minimum_template_margin)
            or not 0 <= minimum_template_margin <= 1
        ):
            raise ValueError("跨模板最佳与次佳差值必须位于 0 到 1 之间")
        self._templates = templates
        self._expected_frame_size = expected_frame_size
        self._minimum_template_margin = minimum_template_margin

    def observe(self, frame: CapturedFrame) -> WaypointObservation:
        actual_size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if actual_size != self._expected_frame_size:
            return WaypointObservation(
                frame_sequence=frame.sequence,
                captured_at_ns=frame.captured_at_ns,
                confidence=0,
                centroid=None,
                waypoint_id=None,
            )
        matches = sorted(
            (
                (route_template, route_template.detector.detect(frame.image))
                for route_template in self._templates
            ),
            key=lambda item: (-item[1].confidence, item[0].template_id),
        )
        best_template, best_match = matches[0]
        runner_up_score = 0 if len(matches) == 1 else matches[1][1].confidence
        accepted = (
            best_match.accepted
            and best_match.confidence - runner_up_score
            >= self._minimum_template_margin
        )
        return WaypointObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            confidence=best_match.confidence,
            centroid=best_template.route_position if accepted else None,
            waypoint_id=best_template.waypoint_id if accepted else None,
        )
