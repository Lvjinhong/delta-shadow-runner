import dataclasses

import cv2
import numpy as np
import pytest

from delta_vision.config import CaptureRegion
from delta_vision.feature_matching import (
    FeatureBackend,
    FeatureMatchPolicy,
    FeatureMatchReason,
    LocalFeatureAnchorDetector,
)

FRAME_SIZE = (720, 500)
DEFAULT_ROI = CaptureRegion(60, 40, 560, 400)


def _template(seed: int = 20260718) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(180, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (231, 171), (255, 255, 255), 3)
    cv2.line(image, (12, 150), (220, 25), (0, 0, 0), 4)
    cv2.circle(image, (55, 55), 24, (255, 255, 255), 4)
    cv2.putText(
        image,
        "DELTA",
        (42, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    return image


def _quad() -> np.ndarray:
    return np.float32([[115, 65], [430, 50], [455, 320], [85, 335]])


def _scene(
    template: np.ndarray,
    *,
    roi: CaptureRegion = DEFAULT_ROI,
    quads: tuple[np.ndarray, ...] = (_quad(),),
) -> np.ndarray:
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    roi_image = frame[roi.top : roi.bottom, roi.left : roi.right]
    source = np.float32(
        [
            [0, 0],
            [template.shape[1] - 1, 0],
            [template.shape[1] - 1, template.shape[0] - 1],
            [0, template.shape[0] - 1],
        ]
    )
    for destination in quads:
        homography = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(template, homography, (roi.width, roi.height))
        mask = cv2.warpPerspective(
            np.full(template.shape[:2], 255, dtype=np.uint8),
            homography,
            (roi.width, roi.height),
        )
        roi_image[mask > 0] = warped[mask > 0]
    return frame


def _policy(**overrides) -> FeatureMatchPolicy:
    values = {
        "ratio_threshold": 0.75,
        "ransac_reprojection_threshold_px": 3.0,
        "minimum_good_matches": 12,
        "minimum_inliers": 10,
        "minimum_inlier_ratio": 0.55,
        "maximum_reprojection_rmse_px": 4.0,
        "maximum_reprojection_p95_px": 6.0,
        "minimum_source_coverage": 0.05,
        "minimum_target_coverage": 0.02,
        "minimum_projected_area_ratio": 0.05,
        "maximum_projected_area_ratio": 0.9,
        "maximum_homography_condition_number": 100_000_000.0,
        "secondary_minimum_inliers": 8,
    }
    values.update(overrides)
    return FeatureMatchPolicy(**values)


def _detector(
    backend: FeatureBackend,
    *,
    template: np.ndarray | None = None,
    roi: CaptureRegion = DEFAULT_ROI,
    policy: FeatureMatchPolicy | None = None,
    maximum_features: int = 1800,
) -> LocalFeatureAnchorDetector:
    return LocalFeatureAnchorDetector(
        label="spawn-a",
        waypoint_id="spawn-a",
        template=_template() if template is None else template,
        search_roi=roi,
        backend=backend,
        policy=_policy() if policy is None else policy,
        maximum_features=maximum_features,
    )


@pytest.mark.parametrize("backend", [FeatureBackend.ORB, FeatureBackend.SIFT])
def test_feature_detector_accepts_perspective_anchor(backend: FeatureBackend) -> None:
    template = _template()
    detector = _detector(backend, template=template)

    observation = detector.detect(_scene(template))

    assert observation.accepted is True
    assert observation.reason is FeatureMatchReason.ACCEPTED
    assert observation.backend is backend
    assert observation.waypoint_id == "spawn-a"
    assert observation.evidence.inlier_count >= 10
    assert observation.evidence.inlier_ratio >= 0.55
    assert observation.evidence.reprojection_rmse_px <= 4
    assert observation.evidence.reprojection_p95_px <= 6
    assert observation.projected_centroid == pytest.approx((335, 232.5), abs=8)


def test_feature_detector_reports_global_coordinates_for_offset_roi() -> None:
    template = _template()
    detector = _detector(FeatureBackend.SIFT, template=template)

    observation = detector.detect(_scene(template))

    assert observation.projected_quad is not None
    assert observation.projected_quad[0] == pytest.approx((175, 105), abs=8)
    assert observation.projected_centroid == pytest.approx((335, 232.5), abs=8)


@pytest.mark.parametrize("backend", [FeatureBackend.ORB, FeatureBackend.SIFT])
def test_feature_detector_rejects_blank_frame_without_descriptors(
    backend: FeatureBackend,
) -> None:
    observation = _detector(backend).detect(
        np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    )

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.NO_FRAME_DESCRIPTORS
    assert observation.projected_quad is None
    assert observation.projected_centroid is None


def test_feature_detector_rejects_unrelated_texture() -> None:
    frame = np.random.default_rng(991).integers(
        0,
        256,
        size=(FRAME_SIZE[1], FRAME_SIZE[0], 3),
        dtype=np.uint8,
    )

    observation = _detector(FeatureBackend.ORB).detect(frame)

    assert observation.accepted is False
    assert observation.reason in {
        FeatureMatchReason.INSUFFICIENT_GOOD_MATCHES,
        FeatureMatchReason.HOMOGRAPHY_FAILED,
        FeatureMatchReason.INSUFFICIENT_INLIERS,
        FeatureMatchReason.LOW_INLIER_RATIO,
    }


def test_feature_detector_rejects_before_homography_when_match_gate_is_impossible() -> None:
    template = _template()
    detector = _detector(
        FeatureBackend.ORB,
        template=template,
        policy=_policy(minimum_good_matches=1000),
    )

    observation = detector.detect(_scene(template))

    assert observation.reason is FeatureMatchReason.INSUFFICIENT_GOOD_MATCHES
    assert observation.evidence.homography is None


def test_feature_detector_rejects_inlier_count_below_policy() -> None:
    template = _template()
    detector = _detector(
        FeatureBackend.SIFT,
        template=template,
        policy=_policy(minimum_inliers=10_000),
    )

    observation = detector.detect(_scene(template))

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.INSUFFICIENT_INLIERS


def test_feature_detector_normalizes_find_homography_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _template()
    detector = _detector(FeatureBackend.SIFT, template=template)

    def fail_homography(*args, **kwargs):
        raise cv2.error("forced RANSAC failure")

    monkeypatch.setattr(cv2, "findHomography", fail_homography)

    observation = detector.detect(_scene(template))

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.HOMOGRAPHY_FAILED


def test_feature_detector_rejects_insufficient_source_coverage() -> None:
    template = _template()
    detector = _detector(
        FeatureBackend.ORB,
        template=template,
        policy=_policy(minimum_source_coverage=0.99),
    )

    observation = detector.detect(_scene(template))

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.LOW_SOURCE_COVERAGE


def test_feature_detector_rejects_projected_quad_outside_search_roi() -> None:
    template = _template()
    outside_quad = np.float32([[-45, 50], [260, 45], [275, 310], [-60, 320]])
    detector = _detector(FeatureBackend.SIFT, template=template)

    observation = detector.detect(_scene(template, quads=(outside_quad,)))

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.INVALID_PROJECTED_QUAD
    assert observation.projected_quad is None


def test_secondary_boundary_tolerance_never_relaxes_primary_acceptance() -> None:
    template = _template()
    slightly_outside = np.float32([[115, 65], [430, 50], [455, 412], [85, 405]])
    detector = _detector(FeatureBackend.SIFT, template=template)

    observation = detector.detect(_scene(template, quads=(slightly_outside,)))

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.INVALID_PROJECTED_QUAD


def test_feature_detector_rejects_two_strong_spatial_models() -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    first = np.float32([[20, 70], [300, 60], [315, 325], [10, 335]])
    second = np.float32([[405, 65], [690, 70], [700, 330], [390, 320]])
    detector = _detector(FeatureBackend.SIFT, template=template, roi=roi)

    observation = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.AMBIGUOUS_SECOND_MODEL
    assert observation.evidence.secondary_inlier_count >= 8


def test_second_model_must_pass_the_same_geometry_gates() -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    first = np.float32([[20, 70], [300, 60], [315, 325], [10, 335]])
    second = np.float32([[405, 65], [690, 70], [700, 330], [390, 320]])
    detector = _detector(
        FeatureBackend.SIFT,
        template=template,
        roi=roi,
        policy=_policy(minimum_source_coverage=0.99),
    )

    observation = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert observation.accepted is False
    assert observation.reason is not FeatureMatchReason.AMBIGUOUS_SECOND_MODEL


def test_repeated_models_remain_ambiguous_after_primary_outlier_split() -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    first = np.float32([[92, 95], [325, 98], [323, 325], [87, 321]])
    second = np.float32([[389, 147], [646, 145], [646, 349], [385, 351]])
    detector = _detector(FeatureBackend.ORB, template=template, roi=roi)

    first_only = detector.detect(_scene(template, roi=roi, quads=(first,)))
    second_only = detector.detect(_scene(template, roi=roi, quads=(second,)))
    combined = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert first_only.reason is FeatureMatchReason.ACCEPTED
    assert second_only.reason is FeatureMatchReason.ACCEPTED
    assert combined.accepted is False
    assert combined.reason is FeatureMatchReason.AMBIGUOUS_SECOND_MODEL


def test_second_model_quality_uses_fail_closed_geometry_not_residual_ratio() -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    first = np.float32([[18, 36], [305, 39], [303, 187], [13, 183]])
    second = np.float32([[386, 151], [591, 149], [591, 324], [382, 326]])
    detector = _detector(FeatureBackend.ORB, template=template, roi=roi)

    first_only = detector.detect(_scene(template, roi=roi, quads=(first,)))
    second_only = detector.detect(_scene(template, roi=roi, quads=(second,)))
    combined = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert first_only.reason is FeatureMatchReason.ACCEPTED
    assert second_only.reason is FeatureMatchReason.ACCEPTED
    assert combined.accepted is False
    assert combined.reason is FeatureMatchReason.AMBIGUOUS_SECOND_MODEL


def test_second_model_allows_bounded_quad_drift_only_for_ambiguity() -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    first = np.float32([[65, 43], [324, 46], [322, 205], [60, 201]])
    second = np.float32([[421, 164], [626, 162], [626, 380], [417, 382]])
    detector = _detector(FeatureBackend.ORB, template=template, roi=roi)

    first_only = detector.detect(_scene(template, roi=roi, quads=(first,)))
    second_only = detector.detect(_scene(template, roi=roi, quads=(second,)))
    combined = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert first_only.reason is FeatureMatchReason.ACCEPTED
    assert second_only.reason is FeatureMatchReason.ACCEPTED
    assert combined.accepted is False
    assert combined.reason is FeatureMatchReason.AMBIGUOUS_SECOND_MODEL


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            np.float32([[26, 140], [318, 143], [316, 326], [21, 322]]),
            np.float32([[425, 180], [625, 178], [625, 384], [421, 386]]),
        ),
        (
            np.float32([[93, 114], [322, 117], [320, 374], [88, 370]]),
            np.float32([[440, 213], [701, 211], [701, 370], [436, 372]]),
        ),
    ],
)
def test_secondary_residual_ratio_and_target_coverage_cannot_hide_duplicate(
    first: np.ndarray,
    second: np.ndarray,
) -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    detector = _detector(FeatureBackend.ORB, template=template, roi=roi)

    first_only = detector.detect(_scene(template, roi=roi, quads=(first,)))
    second_only = detector.detect(_scene(template, roi=roi, quads=(second,)))
    combined = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert first_only.reason is FeatureMatchReason.ACCEPTED
    assert second_only.reason is FeatureMatchReason.ACCEPTED
    assert combined.accepted is False
    assert combined.reason is FeatureMatchReason.AMBIGUOUS_SECOND_MODEL


def test_secondary_residual_source_coverage_cannot_hide_duplicate() -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    first = np.float32([[83, 86], [319, 89], [317, 317], [78, 313]])
    second = np.float32([[397, 102], [595, 100], [595, 312], [393, 314]])
    detector = _detector(FeatureBackend.ORB, template=template, roi=roi)

    first_only = detector.detect(_scene(template, roi=roi, quads=(first,)))
    second_only = detector.detect(_scene(template, roi=roi, quads=(second,)))
    combined = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert first_only.reason is FeatureMatchReason.ACCEPTED
    assert second_only.reason is FeatureMatchReason.ACCEPTED
    assert combined.accepted is False
    assert combined.reason is FeatureMatchReason.AMBIGUOUS_SECOND_MODEL


def test_masked_second_pass_recovers_independently_valid_duplicate() -> None:
    template = _template()
    roi = CaptureRegion(0, 40, 720, 400)
    first = np.float32([[78, 6], [266, 9], [264, 239], [73, 235]])
    second = np.float32([[498, 160], [713, 158], [713, 352], [494, 354]])
    detector = _detector(FeatureBackend.ORB, template=template, roi=roi)

    first_only = detector.detect(_scene(template, roi=roi, quads=(first,)))
    second_only = detector.detect(_scene(template, roi=roi, quads=(second,)))
    combined = detector.detect(_scene(template, roi=roi, quads=(first, second)))

    assert first_only.reason is FeatureMatchReason.ACCEPTED
    assert second_only.reason is FeatureMatchReason.ACCEPTED
    assert combined.accepted is False
    assert combined.reason is FeatureMatchReason.AMBIGUOUS_SECOND_MODEL


def test_orb_and_sift_keep_descriptor_metrics_explicit() -> None:
    template = _template()

    orb = _detector(FeatureBackend.ORB, template=template).detect(_scene(template))
    sift = _detector(FeatureBackend.SIFT, template=template).detect(_scene(template))

    assert orb.evidence.descriptor_metric == "hamming"
    assert sift.evidence.descriptor_metric == "l2"
    assert orb.evidence.mean_match_distance != pytest.approx(sift.evidence.mean_match_distance)
    assert not hasattr(orb, "confidence")


def test_feature_detector_copies_template_and_never_mutates_frame() -> None:
    template = _template()
    original_template = template.copy()
    detector = _detector(FeatureBackend.ORB, template=template)
    template[:] = 0
    frame = _scene(original_template)
    original_frame = frame.copy()

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert first == second
    assert first.accepted is True
    assert np.array_equal(frame, original_frame)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.accepted = False


def test_identity_match_touching_roi_border_tolerates_only_numeric_epsilon() -> None:
    template = _template()
    detector = LocalFeatureAnchorDetector(
        label="full-roi",
        waypoint_id="A",
        template=template,
        search_roi=CaptureRegion(0, 0, template.shape[1], template.shape[0]),
        backend=FeatureBackend.SIFT,
        policy=_policy(
            minimum_projected_area_ratio=0.5,
            maximum_projected_area_ratio=1.0,
            minimum_target_coverage=0.05,
        ),
        maximum_features=2000,
    )

    observation = detector.detect(template)

    assert observation.reason is FeatureMatchReason.ACCEPTED


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((100, 100), dtype=np.uint8),
        np.zeros((100, 100, 3), dtype=np.float32),
        np.zeros((0, 100, 3), dtype=np.uint8),
    ],
)
def test_feature_detector_rejects_invalid_frame(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="帧"):
        _detector(FeatureBackend.ORB).detect(image)


@pytest.mark.parametrize("backend", [FeatureBackend.ORB, FeatureBackend.SIFT])
def test_feature_detector_rejects_template_without_descriptors(
    backend: FeatureBackend,
) -> None:
    with pytest.raises(ValueError, match=r"descriptor|特征"):
        _detector(
            backend,
            template=np.zeros((180, 240, 3), dtype=np.uint8),
        )


def test_feature_detector_rejects_frame_smaller_than_roi() -> None:
    observation = _detector(FeatureBackend.ORB).detect(np.zeros((200, 300, 3), dtype=np.uint8))

    assert observation.accepted is False
    assert observation.reason is FeatureMatchReason.FRAME_SIZE_MISMATCH


@pytest.mark.parametrize("maximum_features", [2_147_483_647, 10**100])
def test_feature_detector_rejects_unsafe_maximum_features(
    maximum_features: int,
) -> None:
    with pytest.raises(ValueError, match="maximum_features"):
        _detector(FeatureBackend.ORB, maximum_features=maximum_features)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ratio_threshold", 1.0),
        ("minimum_good_matches", 3),
        ("minimum_inliers", 3),
        ("minimum_inlier_ratio", 0),
        ("maximum_reprojection_rmse_px", float("inf")),
        ("minimum_source_coverage", 1.1),
        ("minimum_projected_area_ratio", 0),
        ("secondary_maximum_corner_outside_roi_px", -1),
    ],
)
def test_feature_match_policy_rejects_invalid_contract(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _policy(**{field: value})
