import math

import cv2
import numpy as np
import pytest

from delta_vision.config import CaptureRegion
from delta_vision.frames import CapturedFrame
from delta_vision.navigation import ObservationScope
from delta_vision.template_matching import (
    MatchDecisionPolicy,
    RouteTemplate,
    TemplateAnchorDetector,
    TemplateMatchObservation,
    TemplateWaypointObserver,
)

DEFAULT_ROI = CaptureRegion(20, 10, 120, 80)


def _template() -> np.ndarray:
    rng = np.random.default_rng(20260716)
    return rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)


def _detector(
    template: np.ndarray,
    *,
    roi: CaptureRegion = DEFAULT_ROI,
    scales: tuple[float, ...] = (1.0,),
    threshold: float = 0.8,
    margin: float = 0.05,
) -> TemplateAnchorDetector:
    return TemplateAnchorDetector(
        label="minimap-anchor",
        template=template,
        search_roi=roi,
        scales=scales,
        policy=MatchDecisionPolicy(
            score_threshold=threshold,
            minimum_margin=margin,
        ),
        nms_radius_px=18,
    )


def test_template_detector_finds_single_anchor_in_global_coordinates() -> None:
    template = _template()
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[35:47, 60:76] = template
    detector = _detector(template)

    observation = detector.detect(frame)

    assert observation.accepted is True
    assert observation.reason == "accepted"
    assert observation.bbox == (60, 35, 16, 12)
    assert observation.centroid == pytest.approx((68.0, 41.0))
    assert observation.scale == pytest.approx(1.0)
    assert observation.confidence == pytest.approx(1.0, abs=1e-6)


def test_template_detector_ignores_better_match_outside_roi() -> None:
    template = _template()
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    degraded = template.copy()
    degraded[0, 0] = 255 - degraded[0, 0]
    frame[35:47, 60:76] = degraded
    frame[90:102, 150:166] = template
    detector = _detector(template, threshold=0.7)

    observation = detector.detect(frame)

    assert observation.accepted is True
    assert observation.bbox == (60, 35, 16, 12)


def test_template_detector_reports_anchor_touching_roi_edge() -> None:
    template = _template()
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[10:22, 20:36] = template
    detector = _detector(template)

    observation = detector.detect(frame)

    assert observation.bbox == (20, 10, 16, 12)
    assert observation.centroid == pytest.approx((28.0, 16.0))


def test_template_detector_accepts_roi_exactly_the_template_size() -> None:
    template = _template()
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    frame[10:22, 20:36] = template
    detector = _detector(template, roi=CaptureRegion(20, 10, 16, 12))

    observation = detector.detect(frame)

    assert observation.accepted is True
    assert observation.bbox == (20, 10, 16, 12)


def test_template_detector_selects_matching_scale() -> None:
    template = _template()
    scaled = cv2.resize(template, (20, 15), interpolation=cv2.INTER_LINEAR)
    frame = np.zeros((140, 200, 3), dtype=np.uint8)
    frame[40:55, 70:90] = scaled
    detector = _detector(template, scales=(0.8, 1.0, 1.25))

    observation = detector.detect(frame)

    assert observation.accepted is True
    assert observation.bbox == (70, 40, 20, 15)
    assert observation.scale == pytest.approx(1.25)


def test_template_detector_rejects_two_distinct_equal_matches_as_ambiguous() -> None:
    template = _template()
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[25:37, 40:56] = template
    frame[65:77, 100:116] = template
    detector = _detector(template)

    observation = detector.detect(frame)

    assert observation.accepted is False
    assert observation.reason == "ambiguous"
    assert observation.centroid is None
    assert observation.bbox is None
    assert observation.scale is None
    assert observation.candidate_centroid is not None
    assert observation.candidate_bbox is not None
    assert observation.runner_up_confidence == pytest.approx(1.0, abs=1e-6)


def test_template_detector_rejects_low_score_but_keeps_candidate() -> None:
    template = _template()
    rng = np.random.default_rng(99)
    frame = rng.integers(0, 256, size=(120, 180, 3), dtype=np.uint8)
    detector = _detector(template, threshold=0.99)

    observation = detector.detect(frame)

    assert observation.accepted is False
    assert observation.reason == "below_threshold"
    assert observation.centroid is None
    assert observation.candidate_centroid is not None
    assert observation.confidence < 0.99


def test_template_detector_returns_no_candidate_when_every_scale_is_too_large() -> None:
    template = _template()
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    detector = _detector(
        template,
        roi=CaptureRegion(0, 0, 10, 10),
        scales=(1.0, 2.0),
    )

    observation = detector.detect(frame)

    assert observation.reason == "no_candidate"
    assert observation.confidence == 0
    assert observation.centroid is None
    assert observation.candidate_centroid is None


def test_match_policy_has_explicit_inclusive_boundaries() -> None:
    policy = MatchDecisionPolicy(score_threshold=0.8, minimum_margin=0.1)

    assert policy.decide(best_score=0.8, runner_up_score=0.7) == "accepted"
    assert (
        policy.decide(best_score=math.nextafter(0.8, 0), runner_up_score=0.6)
        == "below_threshold"
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_match_policy_rejects_non_finite_runtime_score(score: float) -> None:
    policy = MatchDecisionPolicy(score_threshold=0.8, minimum_margin=0.1)

    with pytest.raises(ValueError, match="有限数"):
        policy.decide(best_score=score, runner_up_score=0.1)
    assert (
        policy.decide(best_score=0.9, runner_up_score=math.nextafter(0.8, 1))
        == "ambiguous"
    )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_template_detector_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="缩放比例"):
        _detector(_template(), scales=(scale,))


def test_template_detector_rejects_constant_template() -> None:
    with pytest.raises(ValueError, match="纹理"):
        _detector(np.zeros((12, 16, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "template",
    [
        np.zeros((12, 16), dtype=np.uint8),
        np.zeros((12, 16, 3), dtype=np.float32),
        np.zeros((0, 16, 3), dtype=np.uint8),
    ],
)
def test_template_detector_rejects_invalid_template_image(template) -> None:
    with pytest.raises(ValueError, match="模板必须"):
        _detector(template)


def test_template_detector_rejects_invalid_constructor_contract() -> None:
    template = _template()
    policy = MatchDecisionPolicy(score_threshold=0.8, minimum_margin=0.1)

    with pytest.raises(ValueError, match="标签"):
        TemplateAnchorDetector(
            label="",
            template=template,
            search_roi=DEFAULT_ROI,
            scales=(1.0,),
            policy=policy,
            nms_radius_px=18,
        )
    with pytest.raises(ValueError, match="左上角"):
        TemplateAnchorDetector(
            label="anchor",
            template=template,
            search_roi=CaptureRegion(-1, 0, 20, 20),
            scales=(1.0,),
            policy=policy,
            nms_radius_px=18,
        )
    with pytest.raises(ValueError, match="至少"):
        TemplateAnchorDetector(
            label="anchor",
            template=template,
            search_roi=DEFAULT_ROI,
            scales=(),
            policy=policy,
            nms_radius_px=18,
        )
    with pytest.raises(ValueError, match="NMS"):
        TemplateAnchorDetector(
            label="anchor",
            template=template,
            search_roi=DEFAULT_ROI,
            scales=(1.0,),
            policy=policy,
            nms_radius_px=0,
        )


def test_template_detector_rejects_roi_outside_frame() -> None:
    detector = _detector(_template(), roi=CaptureRegion(20, 10, 120, 80))

    with pytest.raises(ValueError, match="ROI"):
        detector.detect(np.zeros((50, 80, 3), dtype=np.uint8))


def test_template_detector_rejects_invalid_frame_without_mutating_inputs() -> None:
    template = _template()
    template_before = template.copy()
    detector = _detector(template)
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame_before = frame.copy()

    detector.detect(frame)

    assert np.array_equal(template, template_before)
    assert np.array_equal(frame, frame_before)
    with pytest.raises(ValueError, match="待检测截图"):
        detector.detect(np.zeros((120, 180), dtype=np.uint8))


def _captured_frame(image: np.ndarray, sequence: int = 7) -> CapturedFrame:
    image.setflags(write=False)
    return CapturedFrame(sequence, 1_000_000 + sequence, image, "fixture")


def test_template_waypoint_observer_maps_screen_match_to_route_position() -> None:
    first = _template()
    second = np.random.default_rng(42).integers(
        0, 256, size=first.shape, dtype=np.uint8
    )
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[35:47, 60:76] = first
    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("route-01", _detector(first), (200.0, 10.0), "turn"),
            RouteTemplate("route-02", _detector(second), (300.0, 20.0), None),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )

    observation = observer.observe(
        _captured_frame(frame),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.frame_sequence == 7
    assert observation.confidence == pytest.approx(1.0, abs=1e-6)
    assert observation.centroid == (200.0, 10.0)
    assert observation.waypoint_id == "turn"


def test_template_waypoint_observer_applies_scope_before_margin_without_state() -> None:
    class CountingDetector:
        def __init__(self, confidence: float) -> None:
            self.confidence = confidence
            self.calls = 0

        def detect(self, image: np.ndarray) -> TemplateMatchObservation:
            del image
            self.calls += 1
            accepted = self.confidence >= 0.8
            return TemplateMatchObservation(
                label="counting",
                confidence=self.confidence,
                runner_up_confidence=0,
                bbox=(0, 0, 1, 1) if accepted else None,
                centroid=(0.5, 0.5) if accepted else None,
                scale=1 if accepted else None,
                candidate_bbox=(0, 0, 1, 1),
                candidate_centroid=(0.5, 0.5),
                candidate_scale=1,
                accepted=accepted,
                reason="accepted" if accepted else "below_threshold",
            )

    current = CountingDetector(1)
    next_detector = CountingDetector(0.1)
    distant = CountingDetector(1)
    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("current", current, (0.0, 0.0), "A"),
            RouteTemplate("next", next_detector, (100.0, 0.0), "B"),
            RouteTemplate("distant", distant, (300.0, 0.0), "D"),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )
    frame = _captured_frame(np.zeros((120, 180, 3), dtype=np.uint8))

    first_global = observer.observe(
        frame,
        scope=ObservationScope(allowed_waypoint_ids=None),
    )
    constrained_observation = observer.observe(
        frame,
        scope=ObservationScope(allowed_waypoint_ids=frozenset({"A", "B"})),
    )
    second_global = observer.observe(
        frame,
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert first_global.centroid is None
    assert constrained_observation.waypoint_id == "A"
    assert constrained_observation.centroid == (0.0, 0.0)
    assert second_global.centroid is None
    assert distant.calls == 2


def test_template_waypoint_observer_excludes_unassigned_template_in_route_scope() -> None:
    template = _template()
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[35:47, 60:76] = template
    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("between", _detector(template), (50.0, 0.0), None),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )

    observation = observer.observe(
        _captured_frame(frame),
        scope=ObservationScope(allowed_waypoint_ids=frozenset({"A", "B"})),
    )

    assert observation.confidence == 0
    assert observation.centroid is None
    assert observation.waypoint_id is None


def test_template_waypoint_observer_excludes_unassigned_template_in_global_scope() -> None:
    class MarkerDetector:
        def __init__(self, confidence: float) -> None:
            self.confidence = confidence
            self.calls = 0

        def detect(self, image: np.ndarray) -> TemplateMatchObservation:
            del image
            self.calls += 1
            return TemplateMatchObservation(
                label="marker",
                confidence=self.confidence,
                runner_up_confidence=0,
                bbox=(0, 0, 1, 1),
                centroid=(0.5, 0.5),
                scale=1,
                candidate_bbox=(0, 0, 1, 1),
                candidate_centroid=(0.5, 0.5),
                candidate_scale=1,
                accepted=True,
                reason="accepted",
            )

    known_detector = MarkerDetector(0.9)
    unassigned_detector = MarkerDetector(1.0)
    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("known", known_detector, (10.0, 0.0), "A"),
            RouteTemplate("unassigned", unassigned_detector, (50.0, 0.0), None),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )

    observation = observer.observe(
        _captured_frame(np.zeros((120, 180, 3), dtype=np.uint8)),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.waypoint_id == "A"
    assert observation.centroid == (10.0, 0.0)
    assert known_detector.calls == 1
    assert unassigned_detector.calls == 0


def test_template_waypoint_observer_groups_margin_by_waypoint() -> None:
    class FixedDetector:
        def __init__(self, confidence: float, *, accepted: bool | None = None) -> None:
            self.confidence = confidence
            self.accepted = confidence >= 0.8 if accepted is None else accepted

        def detect(self, image: np.ndarray) -> TemplateMatchObservation:
            del image
            return TemplateMatchObservation(
                label="fixed",
                confidence=self.confidence,
                runner_up_confidence=0,
                bbox=(0, 0, 1, 1) if self.accepted else None,
                centroid=(0.5, 0.5) if self.accepted else None,
                scale=1 if self.accepted else None,
                candidate_bbox=(0, 0, 1, 1),
                candidate_centroid=(0.5, 0.5),
                candidate_scale=1,
                accepted=self.accepted,
                reason="accepted" if self.accepted else "ambiguous",
            )

    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("current", FixedDetector(0.1), (0.0, 0.0), "A"),
            RouteTemplate("next", FixedDetector(0.1), (10.0, 0.0), "B"),
            RouteTemplate("off-route-1", FixedDetector(1.0), (50.0, 0.0), "C"),
            RouteTemplate("off-route-2", FixedDetector(1.0), (50.0, 0.0), "C"),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )

    observation = observer.observe(
        _captured_frame(np.zeros((120, 180, 3), dtype=np.uint8)),
        scope=ObservationScope(allowed_waypoint_ids=frozenset({"A", "B"})),
    )

    assert observation.scope_violation is True
    assert observation.out_of_scope_waypoint_id == "C"

    mixed_observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("current", FixedDetector(0.1), (0.0, 0.0), "A"),
            RouteTemplate(
                "off-route-rejected",
                FixedDetector(0.95, accepted=False),
                (50.0, 0.0),
                "C",
            ),
            RouteTemplate(
                "off-route-accepted",
                FixedDetector(0.9, accepted=True),
                (50.0, 0.0),
                "C",
            ),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )

    mixed_observation = mixed_observer.observe(
        _captured_frame(np.zeros((120, 180, 3), dtype=np.uint8)),
        scope=ObservationScope(allowed_waypoint_ids=frozenset({"A"})),
    )

    assert mixed_observation.scope_violation is True
    assert mixed_observation.out_of_scope_waypoint_id == "C"


def test_template_waypoint_observer_rejects_cross_template_ambiguity() -> None:
    template = _template()
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[35:47, 60:76] = template
    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("left", _detector(template), (100.0, 0.0), "A"),
            RouteTemplate("right", _detector(template), (200.0, 0.0), "B"),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )

    observation = observer.observe(
        _captured_frame(frame),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.confidence == pytest.approx(1.0, abs=1e-6)
    assert observation.centroid is None
    assert observation.waypoint_id is None


def test_template_waypoint_observer_rejects_frame_profile_mismatch() -> None:
    template = _template()
    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("route-01", _detector(template), (100.0, 0.0), "A"),
        ),
        expected_frame_size=(180, 120),
        minimum_template_margin=0.05,
    )

    observation = observer.observe(
        _captured_frame(np.zeros((100, 160, 3), dtype=np.uint8)),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.confidence == 0
    assert observation.centroid is None
    assert observation.waypoint_id is None


def test_route_template_and_observer_reject_invalid_profiles() -> None:
    detector = _detector(_template())
    with pytest.raises(ValueError, match="ID"):
        RouteTemplate("", detector, (1.0, 2.0), None)
    with pytest.raises(ValueError, match="路线坐标"):
        RouteTemplate("bad", detector, (float("nan"), 2.0), None)
    with pytest.raises(ValueError, match="waypoint"):
        RouteTemplate("bad", detector, (1.0, 2.0), "")
    with pytest.raises(ValueError, match="路线模板"):
        TemplateWaypointObserver(
            templates=(),
            expected_frame_size=(180, 120),
            minimum_template_margin=0.05,
        )
    duplicate = RouteTemplate("duplicate", detector, (1.0, 2.0), None)
    with pytest.raises(ValueError, match="不能重复"):
        TemplateWaypointObserver(
            templates=(duplicate, duplicate),
            expected_frame_size=(180, 120),
            minimum_template_margin=0.05,
        )
    with pytest.raises(ValueError, match="分辨率"):
        TemplateWaypointObserver(
            templates=(RouteTemplate("ok", detector, (1.0, 2.0), None),),
            expected_frame_size=(0, 120),
            minimum_template_margin=0.05,
        )
    with pytest.raises(ValueError, match="差值"):
        TemplateWaypointObserver(
            templates=(RouteTemplate("ok", detector, (1.0, 2.0), None),),
            expected_frame_size=(180, 120),
            minimum_template_margin=-0.01,
        )


@pytest.mark.parametrize(
    ("threshold", "margin"),
    [(0, 0.1), (1.01, 0.1), (0.8, -0.01), (0.8, float("nan"))],
)
def test_match_policy_rejects_invalid_thresholds(
    threshold: float, margin: float
) -> None:
    with pytest.raises(ValueError):
        MatchDecisionPolicy(
            score_threshold=threshold,
            minimum_margin=margin,
        )
