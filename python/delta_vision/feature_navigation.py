"""把 ORB/SIFT 几何匹配结果安全映射为路线 waypoint。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from numpy import uint8
from numpy.typing import NDArray

from .feature_matching import FeatureBackend
from .frames import CapturedFrame
from .navigation import ObservationScope, WaypointObservation


class _FeatureEvidence(Protocol):
    raw_quality: float


class _FeatureDetection(Protocol):
    label: str
    waypoint_id: str
    backend: FeatureBackend
    accepted: bool
    evidence: _FeatureEvidence


class FeatureAnchorDetector(Protocol):
    @property
    def label(self) -> str: ...

    @property
    def waypoint_id(self) -> str: ...

    @property
    def backend(self) -> FeatureBackend: ...

    def detect(self, frame: NDArray[uint8]) -> _FeatureDetection: ...


@dataclass(frozen=True, slots=True)
class FeatureRouteTemplate:
    detector: FeatureAnchorDetector
    route_position: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.detector.label, str) or not self.detector.label:
            raise ValueError("特征路线模板 label 不能为空")
        if not isinstance(self.detector.waypoint_id, str) or not self.detector.waypoint_id:
            raise ValueError("特征路线模板 waypoint ID 不能为空")
        if not isinstance(self.detector.backend, FeatureBackend):
            raise ValueError("特征路线模板 backend 无效")
        if len(self.route_position) != 2 or not all(
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            for value in self.route_position
        ):
            raise ValueError("特征路线坐标必须包含两个有限数")


@dataclass(frozen=True, slots=True)
class _OwnedFeatureRouteTemplate:
    detector: FeatureAnchorDetector
    label: str
    waypoint_id: str
    backend: FeatureBackend
    route_position: tuple[float, float]


class FeatureWaypointObserver:
    """只在一个唯一 waypoint 的几何模型通过时输出路线锁定。"""

    def __init__(
        self,
        *,
        templates: tuple[FeatureRouteTemplate, ...],
        expected_frame_size: tuple[int, int],
    ) -> None:
        supplied_templates = tuple(templates)
        if not supplied_templates:
            raise ValueError("特征路线模板不能为空")
        if any(not isinstance(template, FeatureRouteTemplate) for template in supplied_templates):
            raise ValueError("特征路线模板类型无效")
        templates = tuple(
            _OwnedFeatureRouteTemplate(
                detector=template.detector,
                label=template.detector.label,
                waypoint_id=template.detector.waypoint_id,
                backend=template.detector.backend,
                route_position=tuple(template.route_position),
            )
            for template in supplied_templates
        )
        labels = tuple(template.label for template in templates)
        if len(labels) != len(set(labels)):
            raise ValueError("特征路线模板 label 不能重复")
        backends = {template.backend for template in templates}
        if len(backends) != 1:
            raise ValueError("同一 Observer 的特征 backend 必须一致")
        if len(expected_frame_size) != 2 or any(
            type(value) is not int or value <= 0 for value in expected_frame_size
        ):
            raise ValueError("期望帧分辨率必须包含两个正整数")
        positions_by_waypoint: dict[str, tuple[float, float]] = {}
        for template in templates:
            waypoint_id = template.waypoint_id
            existing = positions_by_waypoint.setdefault(
                waypoint_id,
                template.route_position,
            )
            if existing != template.route_position:
                raise ValueError("同一 waypoint 的特征路线坐标必须一致")
        self._templates = templates
        self._expected_frame_size = expected_frame_size

    @staticmethod
    def _empty_observation(
        frame: CapturedFrame,
        *,
        confidence: float = 0,
    ) -> WaypointObservation:
        return WaypointObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            confidence=confidence,
            centroid=None,
            waypoint_id=None,
        )

    @staticmethod
    def _best_by_waypoint(
        frame: CapturedFrame,
        templates: tuple[_OwnedFeatureRouteTemplate, ...],
    ) -> Mapping[str, tuple[_OwnedFeatureRouteTemplate, float]]:
        accepted: dict[str, tuple[_OwnedFeatureRouteTemplate, float]] = {}
        for template in templates:
            detection = template.detector.detect(frame.image)
            if (
                detection.label != template.label
                or detection.waypoint_id != template.waypoint_id
                or detection.backend is not template.backend
            ):
                raise RuntimeError("特征 detector 返回值违反模板契约")
            if type(detection.accepted) is not bool:
                raise RuntimeError("特征 detector 的 accepted 必须是布尔值")
            quality = detection.evidence.raw_quality
            if (
                isinstance(quality, bool)
                or not isinstance(quality, (int, float))
                or not math.isfinite(quality)
                or not 0 <= quality <= 1
            ):
                raise RuntimeError("特征 detector 返回了无效质量分数")
            if not detection.accepted:
                continue
            existing = accepted.get(detection.waypoint_id)
            if existing is None or quality > existing[1]:
                accepted[detection.waypoint_id] = (template, float(quality))
        return accepted

    def observe(
        self,
        frame: CapturedFrame,
        *,
        scope: ObservationScope,
    ) -> WaypointObservation:
        actual_size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if actual_size != self._expected_frame_size:
            return self._empty_observation(frame)
        allowed = scope.allowed_waypoint_ids
        normal_templates = (
            self._templates
            if allowed is None
            else tuple(template for template in self._templates if template.waypoint_id in allowed)
        )
        normal = self._best_by_waypoint(frame, normal_templates)
        if len(normal) == 1:
            waypoint_id, (template, quality) = next(iter(normal.items()))
            return WaypointObservation(
                frame_sequence=frame.sequence,
                captured_at_ns=frame.captured_at_ns,
                confidence=quality,
                centroid=template.route_position,
                waypoint_id=waypoint_id,
            )

        confidence = max((match[1] for match in normal.values()), default=0)
        if len(normal) > 1:
            # 当前/下一节点自身存在歧义时保持 uncertain，不能再用路线外命中
            # 覆盖这份证据并把它误报为偏航。
            return self._empty_observation(frame, confidence=confidence)
        if allowed:
            excluded_templates = tuple(
                template for template in self._templates if template.waypoint_id not in allowed
            )
            excluded = self._best_by_waypoint(frame, excluded_templates)
            if len(excluded) == 1:
                waypoint_id, (_, quality) = next(iter(excluded.items()))
                return WaypointObservation(
                    frame_sequence=frame.sequence,
                    captured_at_ns=frame.captured_at_ns,
                    confidence=quality,
                    centroid=None,
                    waypoint_id=None,
                    out_of_scope_waypoint_id=waypoint_id,
                    scope_violation=True,
                )
            confidence = max(
                confidence,
                max((match[1] for match in excluded.values()), default=0),
            )
        return self._empty_observation(frame, confidence=confidence)
