"""ORB/SIFT 局部特征匹配与 fail-closed 几何验证。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import CaptureRegion


class FeatureBackend(StrEnum):
    ORB = "orb"
    SIFT = "sift"


class FeatureMatchReason(StrEnum):
    ACCEPTED = "accepted"
    FRAME_SIZE_MISMATCH = "frame_size_mismatch"
    NO_FRAME_DESCRIPTORS = "no_frame_descriptors"
    INSUFFICIENT_GOOD_MATCHES = "insufficient_good_matches"
    HOMOGRAPHY_FAILED = "homography_failed"
    INVALID_HOMOGRAPHY = "invalid_homography"
    INSUFFICIENT_INLIERS = "insufficient_inliers"
    LOW_INLIER_RATIO = "low_inlier_ratio"
    HIGH_REPROJECTION_ERROR = "high_reprojection_error"
    LOW_SOURCE_COVERAGE = "low_source_coverage"
    LOW_TARGET_COVERAGE = "low_target_coverage"
    INVALID_PROJECTED_QUAD = "invalid_projected_quad"
    AMBIGUOUS_SECOND_MODEL = "ambiguous_second_model"


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class FeatureMatchPolicy:
    ratio_threshold: float
    ransac_reprojection_threshold_px: float
    minimum_good_matches: int
    minimum_inliers: int
    minimum_inlier_ratio: float
    maximum_reprojection_rmse_px: float
    maximum_reprojection_p95_px: float
    minimum_source_coverage: float
    minimum_target_coverage: float
    minimum_projected_area_ratio: float
    maximum_projected_area_ratio: float
    maximum_homography_condition_number: float
    secondary_minimum_inliers: int
    minimum_projected_edge_px: float = 8.0
    secondary_maximum_corner_outside_roi_px: float = 24.0

    def __post_init__(self) -> None:
        if not _finite(self.ratio_threshold) or not 0 < self.ratio_threshold < 1:
            raise ValueError("ratio_threshold 必须位于 0 到 1 之间")
        positive_numbers = {
            "ransac_reprojection_threshold_px": self.ransac_reprojection_threshold_px,
            "maximum_reprojection_rmse_px": self.maximum_reprojection_rmse_px,
            "maximum_reprojection_p95_px": self.maximum_reprojection_p95_px,
            "maximum_homography_condition_number": (self.maximum_homography_condition_number),
            "minimum_projected_edge_px": self.minimum_projected_edge_px,
        }
        if any(not _finite(value) or value <= 0 for value in positive_numbers.values()):
            raise ValueError("几何阈值必须是正有限数")
        if (
            type(self.minimum_good_matches) is not int
            or self.minimum_good_matches < 4
            or type(self.minimum_inliers) is not int
            or self.minimum_inliers < 4
            or type(self.secondary_minimum_inliers) is not int
            or self.secondary_minimum_inliers < 4
        ):
            raise ValueError("匹配数和 inlier 数下限必须是大于等于 4 的整数")
        unit_interval = (
            self.minimum_inlier_ratio,
            self.minimum_source_coverage,
            self.minimum_target_coverage,
            self.minimum_projected_area_ratio,
            self.maximum_projected_area_ratio,
        )
        if any(not _finite(value) or not 0 < value <= 1 for value in unit_interval):
            raise ValueError("比例阈值必须位于 0 到 1 之间")
        if self.maximum_projected_area_ratio <= self.minimum_projected_area_ratio:
            raise ValueError("投影面积上限必须大于下限")
        if (
            not _finite(self.secondary_maximum_corner_outside_roi_px)
            or self.secondary_maximum_corner_outside_roi_px < 0
        ):
            raise ValueError("secondary 最大越界容差必须是非负有限数")


HomographyTuple = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
QuadTuple = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


@dataclass(frozen=True, slots=True)
class FeatureMatchEvidence:
    descriptor_metric: str
    template_keypoint_count: int
    frame_keypoint_count: int
    knn_pair_count: int
    good_match_count: int
    inlier_count: int = 0
    inlier_ratio: float = 0
    reprojection_rmse_px: float = math.inf
    reprojection_p95_px: float = math.inf
    source_coverage: float = 0
    target_coverage: float = 0
    homography_condition_number: float = math.inf
    mean_match_distance: float = math.inf
    raw_quality: float = 0
    homography: HomographyTuple | None = None
    candidate_projected_quad: QuadTuple | None = None
    secondary_inlier_count: int = 0
    secondary_raw_quality: float = 0


@dataclass(frozen=True, slots=True)
class FeatureMatchObservation:
    label: str
    waypoint_id: str
    backend: FeatureBackend
    accepted: bool
    reason: FeatureMatchReason
    projected_quad: QuadTuple | None
    projected_centroid: tuple[float, float] | None
    evidence: FeatureMatchEvidence


@dataclass(frozen=True, slots=True)
class _Geometry:
    homography: NDArray[np.float64]
    mask: NDArray[np.bool_]
    inlier_count: int
    inlier_ratio: float
    rmse_px: float
    p95_px: float
    source_coverage: float
    target_coverage: float
    condition_number: float
    local_quad: NDArray[np.float32]
    raw_quality: float


class LocalFeatureAnchorDetector:
    """在固定 ROI 中查找一个局部特征锚点，并输出可解释几何证据。"""

    def __init__(
        self,
        *,
        label: str,
        waypoint_id: str,
        template: NDArray[np.uint8],
        search_roi: CaptureRegion,
        backend: FeatureBackend,
        policy: FeatureMatchPolicy,
        maximum_features: int,
    ) -> None:
        if not isinstance(label, str) or not label:
            raise ValueError("特征模板 label 不能为空")
        if not isinstance(waypoint_id, str) or not waypoint_id:
            raise ValueError("特征模板 waypoint_id 不能为空")
        self._validate_image(template, field="模板")
        if search_roi.left < 0 or search_roi.top < 0:
            raise ValueError("搜索 ROI 左上角不能为负数")
        if not isinstance(backend, FeatureBackend):
            raise ValueError("backend 必须是 FeatureBackend")
        if not isinstance(policy, FeatureMatchPolicy):
            raise ValueError("policy 必须是 FeatureMatchPolicy")
        if type(maximum_features) is not int or not 32 <= maximum_features <= 50_000:
            raise ValueError("maximum_features 必须是 32 到 50000 的整数")

        owned_template = np.array(template[:, :, :3], dtype=np.uint8, copy=True)
        gray_template = cv2.cvtColor(owned_template, cv2.COLOR_BGR2GRAY)
        extractor = self._create_extractor(backend, maximum_features)
        template_keypoints, template_descriptors = extractor.detectAndCompute(
            gray_template,
            None,
        )
        if template_descriptors is None or len(template_keypoints) < policy.minimum_good_matches:
            raise ValueError("模板没有足够的局部特征 descriptor")

        owned_template.setflags(write=False)
        gray_template.setflags(write=False)
        template_descriptors = np.array(template_descriptors, copy=True)
        template_descriptors.setflags(write=False)
        self._label = label
        self._waypoint_id = waypoint_id
        self._template = owned_template
        self._template_gray = gray_template
        self._template_keypoints = tuple(template_keypoints)
        self._template_descriptors = template_descriptors
        self._search_roi = search_roi
        self._backend = backend
        self._policy = policy
        self._maximum_features = maximum_features
        self._descriptor_metric = "hamming" if backend is FeatureBackend.ORB else "l2"
        self._norm = cv2.NORM_HAMMING if backend is FeatureBackend.ORB else cv2.NORM_L2

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
    def _create_extractor(backend: FeatureBackend, maximum_features: int):
        if backend is FeatureBackend.ORB:
            return cv2.ORB_create(nfeatures=maximum_features)
        return cv2.SIFT_create(nfeatures=maximum_features)

    def _evidence(
        self,
        *,
        frame_keypoints: int,
        knn_pairs: int,
        good_matches: int,
        mean_distance: float = math.inf,
        geometry: _Geometry | None = None,
        secondary_inliers: int = 0,
        secondary_quality: float = 0,
    ) -> FeatureMatchEvidence:
        homography = None
        quad = None
        if geometry is not None:
            homography = tuple(tuple(float(value) for value in row) for row in geometry.homography)
            quad = tuple((float(point[0]), float(point[1])) for point in geometry.local_quad)
        return FeatureMatchEvidence(
            descriptor_metric=self._descriptor_metric,
            template_keypoint_count=len(self._template_keypoints),
            frame_keypoint_count=frame_keypoints,
            knn_pair_count=knn_pairs,
            good_match_count=good_matches,
            inlier_count=0 if geometry is None else geometry.inlier_count,
            inlier_ratio=0 if geometry is None else geometry.inlier_ratio,
            reprojection_rmse_px=math.inf if geometry is None else geometry.rmse_px,
            reprojection_p95_px=math.inf if geometry is None else geometry.p95_px,
            source_coverage=0 if geometry is None else geometry.source_coverage,
            target_coverage=0 if geometry is None else geometry.target_coverage,
            homography_condition_number=(
                math.inf if geometry is None else geometry.condition_number
            ),
            mean_match_distance=mean_distance,
            raw_quality=0 if geometry is None else geometry.raw_quality,
            homography=homography,
            candidate_projected_quad=quad,
            secondary_inlier_count=secondary_inliers,
            secondary_raw_quality=secondary_quality,
        )

    def _observation(
        self,
        reason: FeatureMatchReason,
        evidence: FeatureMatchEvidence,
        *,
        geometry: _Geometry | None = None,
    ) -> FeatureMatchObservation:
        accepted = reason is FeatureMatchReason.ACCEPTED
        global_quad: QuadTuple | None = None
        centroid: tuple[float, float] | None = None
        if accepted and geometry is not None:
            global_quad = tuple(
                (
                    float(point[0] + self._search_roi.left),
                    float(point[1] + self._search_roi.top),
                )
                for point in geometry.local_quad
            )
            centroid = (
                float(sum(point[0] for point in global_quad) / 4),
                float(sum(point[1] for point in global_quad) / 4),
            )
        return FeatureMatchObservation(
            label=self._label,
            waypoint_id=self._waypoint_id,
            backend=self._backend,
            accepted=accepted,
            reason=reason,
            projected_quad=global_quad,
            projected_centroid=centroid,
            evidence=evidence,
        )

    @staticmethod
    def _coverage(points: NDArray[np.float32], total_area: float) -> float:
        if len(points) < 3 or total_area <= 0:
            return 0
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        return max(0.0, min(1.0, float(cv2.contourArea(hull)) / total_area))

    def _quad_is_valid(
        self,
        quad: NDArray[np.float32],
        *,
        maximum_outside_px: float = 0,
    ) -> bool:
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            return False
        contour = quad.reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour):
            return False
        signed_area = float(cv2.contourArea(contour, oriented=True))
        roi_area = float(self._search_roi.width * self._search_roi.height)
        area_ratio = signed_area / roi_area
        if not (
            self._policy.minimum_projected_area_ratio
            <= area_ratio
            <= self._policy.maximum_projected_area_ratio
        ):
            return False
        edges = tuple(
            float(np.linalg.norm(quad[(index + 1) % 4] - quad[index])) for index in range(4)
        )
        if min(edges) < self._policy.minimum_projected_edge_px:
            return False
        return bool(
            np.all(quad[:, 0] >= -maximum_outside_px)
            and np.all(quad[:, 1] >= -maximum_outside_px)
            and np.all(quad[:, 0] <= self._search_roi.width + maximum_outside_px)
            and np.all(quad[:, 1] <= self._search_roi.height + maximum_outside_px)
        )

    def _geometry(
        self,
        source_points: NDArray[np.float32],
        target_points: NDArray[np.float32],
    ) -> _Geometry | None:
        try:
            homography, raw_mask = cv2.findHomography(
                source_points,
                target_points,
                cv2.RANSAC,
                self._policy.ransac_reprojection_threshold_px,
                maxIters=2000,
                confidence=0.995,
            )
        except cv2.error:
            return None
        if homography is None or raw_mask is None:
            return None
        homography = np.asarray(homography, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            return None
        if abs(float(homography[2, 2])) < np.finfo(np.float64).eps:
            return None
        homography = homography / homography[2, 2]
        condition_number = float(np.linalg.cond(homography))
        mask = np.asarray(raw_mask, dtype=np.uint8).reshape(-1).astype(bool)
        if len(mask) != len(source_points):
            return None
        inlier_count = int(np.count_nonzero(mask))
        inlier_ratio = inlier_count / len(source_points)
        inlier_source = source_points[mask]
        inlier_target = target_points[mask]
        if inlier_count:
            projected = cv2.perspectiveTransform(
                inlier_source.reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2)
            errors = np.linalg.norm(projected - inlier_target, axis=1)
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            p95 = float(np.percentile(errors, 95))
        else:
            rmse = math.inf
            p95 = math.inf
        template_height, template_width = self._template_gray.shape
        source_coverage = self._coverage(
            inlier_source,
            float(template_width * template_height),
        )
        target_coverage = self._coverage(
            inlier_target,
            float(self._search_roi.width * self._search_roi.height),
        )
        corners = np.float32(
            [
                [0, 0],
                [template_width - 1, 0],
                [template_width - 1, template_height - 1],
                [0, template_height - 1],
            ]
        )
        quad = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
        coverage_factor = min(
            1.0,
            source_coverage / self._policy.minimum_source_coverage,
            target_coverage / self._policy.minimum_target_coverage,
        )
        error_factor = (
            0
            if not math.isfinite(rmse)
            else math.exp(-rmse / self._policy.maximum_reprojection_rmse_px)
        )
        raw_quality = inlier_ratio * coverage_factor * error_factor
        homography.setflags(write=False)
        mask.setflags(write=False)
        quad = np.asarray(quad, dtype=np.float32)
        quad.setflags(write=False)
        return _Geometry(
            homography=homography,
            mask=mask,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_px=rmse,
            p95_px=p95,
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            condition_number=condition_number,
            local_quad=quad,
            raw_quality=raw_quality,
        )

    def _rejection_reason(self, geometry: _Geometry) -> FeatureMatchReason | None:
        if (
            not math.isfinite(geometry.condition_number)
            or geometry.condition_number > self._policy.maximum_homography_condition_number
        ):
            return FeatureMatchReason.INVALID_HOMOGRAPHY
        if geometry.inlier_count < self._policy.minimum_inliers:
            return FeatureMatchReason.INSUFFICIENT_INLIERS
        if geometry.inlier_ratio < self._policy.minimum_inlier_ratio:
            return FeatureMatchReason.LOW_INLIER_RATIO
        if (
            geometry.rmse_px > self._policy.maximum_reprojection_rmse_px
            or geometry.p95_px > self._policy.maximum_reprojection_p95_px
        ):
            return FeatureMatchReason.HIGH_REPROJECTION_ERROR
        if geometry.source_coverage < self._policy.minimum_source_coverage:
            return FeatureMatchReason.LOW_SOURCE_COVERAGE
        if geometry.target_coverage < self._policy.minimum_target_coverage:
            return FeatureMatchReason.LOW_TARGET_COVERAGE
        if not self._quad_is_valid(geometry.local_quad):
            return FeatureMatchReason.INVALID_PROJECTED_QUAD
        return None

    def _passes_geometry_gates(
        self,
        geometry: _Geometry,
        *,
        minimum_inliers: int,
        minimum_inlier_ratio: float | None,
        minimum_source_coverage: float | None,
        minimum_target_coverage: float | None,
        maximum_corner_outside_roi_px: float = 0,
    ) -> bool:
        return bool(
            math.isfinite(geometry.condition_number)
            and geometry.condition_number <= self._policy.maximum_homography_condition_number
            and geometry.inlier_count >= minimum_inliers
            and (minimum_inlier_ratio is None or geometry.inlier_ratio >= minimum_inlier_ratio)
            and geometry.rmse_px <= self._policy.maximum_reprojection_rmse_px
            and geometry.p95_px <= self._policy.maximum_reprojection_p95_px
            and (
                minimum_source_coverage is None
                or geometry.source_coverage >= minimum_source_coverage
            )
            and (
                minimum_target_coverage is None
                or geometry.target_coverage >= minimum_target_coverage
            )
            and self._quad_is_valid(
                geometry.local_quad,
                maximum_outside_px=maximum_corner_outside_roi_px,
            )
        )

    def _secondary_model(
        self,
        source_points: NDArray[np.float32],
        target_points: NDArray[np.float32],
        primary: _Geometry,
    ) -> _Geometry | None:
        remaining = ~primary.mask
        if int(np.count_nonzero(remaining)) < self._policy.secondary_minimum_inliers:
            return None
        secondary = self._geometry(source_points[remaining], target_points[remaining])
        if secondary is None:
            return None
        if not self._passes_geometry_gates(
            secondary,
            minimum_inliers=self._policy.secondary_minimum_inliers,
            minimum_inlier_ratio=None,
            minimum_source_coverage=None,
            minimum_target_coverage=None,
            maximum_corner_outside_roi_px=(self._policy.secondary_maximum_corner_outside_roi_px),
        ):
            return None
        if not self._models_are_separated(primary, secondary):
            return None
        return secondary

    def _models_are_separated(
        self,
        primary: _Geometry,
        secondary: _Geometry,
    ) -> bool:
        primary_center = np.mean(primary.local_quad, axis=0)
        secondary_center = np.mean(secondary.local_quad, axis=0)
        minimum_separation = 0.1 * math.hypot(
            self._search_roi.width,
            self._search_roi.height,
        )
        return bool(float(np.linalg.norm(primary_center - secondary_center)) >= minimum_separation)

    def _match_descriptors(
        self,
        frame_descriptors: NDArray,
    ) -> tuple[tuple[tuple[cv2.DMatch, ...], ...], tuple[cv2.DMatch, ...], float]:
        matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        raw_pairs = tuple(
            tuple(pair)
            for pair in matcher.knnMatch(
                frame_descriptors,
                self._template_descriptors,
                k=2,
            )
        )
        good_matches = tuple(
            pair[0]
            for pair in raw_pairs
            if len(pair) == 2 and pair[0].distance < self._policy.ratio_threshold * pair[1].distance
        )
        mean_distance = (
            math.inf
            if not good_matches
            else float(np.mean([match.distance for match in good_matches]))
        )
        return raw_pairs, good_matches, mean_distance

    def _masked_secondary_model(
        self,
        gray: NDArray[np.uint8],
        primary: _Geometry,
    ) -> _Geometry | None:
        feature_mask = np.full(gray.shape, 255, dtype=np.uint8)
        polygon = np.rint(primary.local_quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(feature_mask, polygon, 0)
        extractor = self._create_extractor(self._backend, self._maximum_features)
        keypoints, descriptors = extractor.detectAndCompute(gray, feature_mask)
        if descriptors is None or not keypoints:
            return None
        _raw_pairs, good_matches, _mean_distance = self._match_descriptors(descriptors)
        if len(good_matches) < self._policy.minimum_good_matches:
            return None
        source_points = np.float32(
            [self._template_keypoints[match.trainIdx].pt for match in good_matches]
        )
        target_points = np.float32([keypoints[match.queryIdx].pt for match in good_matches])
        secondary = self._geometry(source_points, target_points)
        if secondary is None or not self._passes_geometry_gates(
            secondary,
            minimum_inliers=self._policy.minimum_inliers,
            minimum_inlier_ratio=None,
            minimum_source_coverage=None,
            minimum_target_coverage=None,
            maximum_corner_outside_roi_px=(self._policy.secondary_maximum_corner_outside_roi_px),
        ):
            return None
        return secondary if self._models_are_separated(primary, secondary) else None

    def detect(self, frame: NDArray[np.uint8]) -> FeatureMatchObservation:
        self._validate_image(frame, field="帧")
        if self._search_roi.right > frame.shape[1] or self._search_roi.bottom > frame.shape[0]:
            return self._observation(
                FeatureMatchReason.FRAME_SIZE_MISMATCH,
                self._evidence(frame_keypoints=0, knn_pairs=0, good_matches=0),
            )
        roi = frame[
            self._search_roi.top : self._search_roi.bottom,
            self._search_roi.left : self._search_roi.right,
            :3,
        ]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        extractor = self._create_extractor(self._backend, self._maximum_features)
        frame_keypoints, frame_descriptors = extractor.detectAndCompute(gray, None)
        if frame_descriptors is None or not frame_keypoints:
            return self._observation(
                FeatureMatchReason.NO_FRAME_DESCRIPTORS,
                self._evidence(frame_keypoints=0, knn_pairs=0, good_matches=0),
            )
        raw_pairs, good_matches, mean_distance = self._match_descriptors(frame_descriptors)
        base_evidence = self._evidence(
            frame_keypoints=len(frame_keypoints),
            knn_pairs=len(raw_pairs),
            good_matches=len(good_matches),
            mean_distance=mean_distance,
        )
        if len(good_matches) < self._policy.minimum_good_matches:
            return self._observation(
                FeatureMatchReason.INSUFFICIENT_GOOD_MATCHES,
                base_evidence,
            )
        source_points = np.float32(
            [self._template_keypoints[match.trainIdx].pt for match in good_matches]
        )
        target_points = np.float32([frame_keypoints[match.queryIdx].pt for match in good_matches])
        primary = self._geometry(source_points, target_points)
        if primary is None:
            return self._observation(
                FeatureMatchReason.HOMOGRAPHY_FAILED,
                base_evidence,
            )
        evidence = self._evidence(
            frame_keypoints=len(frame_keypoints),
            knn_pairs=len(raw_pairs),
            good_matches=len(good_matches),
            mean_distance=mean_distance,
            geometry=primary,
        )
        rejection = self._rejection_reason(primary)
        secondary = None
        if self._passes_geometry_gates(
            primary,
            minimum_inliers=self._policy.minimum_inliers,
            minimum_inlier_ratio=None,
            minimum_source_coverage=self._policy.minimum_source_coverage,
            minimum_target_coverage=self._policy.minimum_target_coverage,
        ):
            secondary = self._secondary_model(source_points, target_points, primary)
            if secondary is None:
                secondary = self._masked_secondary_model(gray, primary)
        if secondary is not None:
            ambiguous_evidence = self._evidence(
                frame_keypoints=len(frame_keypoints),
                knn_pairs=len(raw_pairs),
                good_matches=len(good_matches),
                mean_distance=mean_distance,
                geometry=primary,
                secondary_inliers=secondary.inlier_count,
                secondary_quality=secondary.raw_quality,
            )
            return self._observation(
                FeatureMatchReason.AMBIGUOUS_SECOND_MODEL,
                ambiguous_evidence,
            )
        if rejection is not None:
            return self._observation(rejection, evidence)
        return self._observation(
            FeatureMatchReason.ACCEPTED,
            evidence,
            geometry=primary,
        )
