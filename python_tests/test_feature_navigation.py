from dataclasses import dataclass

import numpy as np
import pytest

from delta_vision.feature_matching import FeatureBackend
from delta_vision.feature_navigation import FeatureRouteTemplate, FeatureWaypointObserver
from delta_vision.frames import CapturedFrame
from delta_vision.navigation import ObservationScope


@dataclass(frozen=True)
class _Evidence:
    raw_quality: float


@dataclass(frozen=True)
class _Detection:
    label: str
    waypoint_id: str
    backend: FeatureBackend
    accepted: bool
    evidence: _Evidence


class _Detector:
    def __init__(
        self,
        *,
        label: str,
        waypoint_id: str,
        accepted_codes: set[int],
        quality: float,
        backend: FeatureBackend = FeatureBackend.SIFT,
        returned_label: str | None = None,
        accepted_value=None,
    ) -> None:
        self.label = label
        self.waypoint_id = waypoint_id
        self.backend = backend
        self._accepted_codes = frozenset(accepted_codes)
        self._quality = quality
        self._returned_label = returned_label or label
        self._accepted_value = accepted_value

    def detect(self, frame):
        code = int(frame[0, 0, 0])
        return _Detection(
            label=self._returned_label,
            waypoint_id=self.waypoint_id,
            backend=self.backend,
            accepted=(
                code in self._accepted_codes
                if self._accepted_value is None
                else self._accepted_value
            ),
            evidence=_Evidence(raw_quality=self._quality),
        )


def _template(
    label: str,
    waypoint_id: str,
    position: tuple[float, float],
    *,
    codes: set[int],
    quality: float = 0.8,
    backend: FeatureBackend = FeatureBackend.SIFT,
) -> FeatureRouteTemplate:
    return FeatureRouteTemplate(
        detector=_Detector(
            label=label,
            waypoint_id=waypoint_id,
            accepted_codes=codes,
            quality=quality,
            backend=backend,
        ),
        route_position=position,
    )


def _frame(sequence: int, code: int, *, size: tuple[int, int] = (80, 60)) -> CapturedFrame:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[0, 0, 0] = code
    image.setflags(write=False)
    return CapturedFrame(sequence, 1_000 + sequence, image, "fixture")


def _observer(*templates: FeatureRouteTemplate) -> FeatureWaypointObserver:
    return FeatureWaypointObserver(
        templates=tuple(templates),
        expected_frame_size=(80, 60),
    )


def test_unique_accepted_feature_waypoint_maps_to_route_position() -> None:
    observer = _observer(
        _template("a", "A", (10, 20), codes={1}, quality=0.72),
        _template("b", "B", (30, 40), codes={2}),
    )

    observation = observer.observe(
        _frame(0, 1),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.waypoint_id == "A"
    assert observation.centroid == (10, 20)
    assert observation.confidence == pytest.approx(0.72)


def test_two_accepted_feature_waypoints_fail_closed_as_ambiguous() -> None:
    observer = _observer(
        _template("a", "A", (10, 20), codes={1}),
        _template("b", "B", (30, 40), codes={1}),
    )

    observation = observer.observe(
        _frame(0, 1),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.waypoint_id is None
    assert observation.centroid is None
    assert observation.confidence == pytest.approx(0.8)


def test_multiple_appearances_of_same_waypoint_do_not_create_false_ambiguity() -> None:
    observer = _observer(
        _template("a-day", "A", (10, 20), codes={1}, quality=0.7),
        _template("a-night", "A", (10, 20), codes={1}, quality=0.9),
        _template("b", "B", (30, 40), codes={2}),
    )

    observation = observer.observe(
        _frame(0, 1),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.waypoint_id == "A"
    assert observation.confidence == pytest.approx(0.9)


def test_excluded_feature_waypoint_is_only_diagnostic_after_normal_rejection() -> None:
    observer = _observer(
        _template("a", "A", (10, 20), codes={1}),
        _template("c", "C", (50, 40), codes={2}),
    )
    scope = ObservationScope(allowed_waypoint_ids=frozenset({"A"}))

    normal = observer.observe(_frame(0, 1), scope=scope)
    excluded = observer.observe(_frame(1, 2), scope=scope)

    assert normal.waypoint_id == "A"
    assert normal.scope_violation is False
    assert excluded.waypoint_id is None
    assert excluded.out_of_scope_waypoint_id == "C"
    assert excluded.scope_violation is True


def test_ambiguous_normal_candidates_do_not_escalate_to_excluded_waypoint() -> None:
    observer = _observer(
        _template("a", "A", (10, 20), codes={1}, quality=0.7),
        _template("b", "B", (30, 40), codes={1}, quality=0.8),
        _template("c", "C", (50, 40), codes={1}, quality=0.9),
    )

    observation = observer.observe(
        _frame(0, 1),
        scope=ObservationScope(allowed_waypoint_ids=frozenset({"A", "B"})),
    )

    assert observation.waypoint_id is None
    assert observation.centroid is None
    assert observation.scope_violation is False
    assert observation.out_of_scope_waypoint_id is None
    assert observation.confidence == pytest.approx(0.8)


def test_feature_observer_rejects_wrong_frame_size_without_running_route_lock() -> None:
    observer = _observer(_template("a", "A", (10, 20), codes={1}))

    observation = observer.observe(
        _frame(0, 1, size=(79, 60)),
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert observation.centroid is None
    assert observation.waypoint_id is None
    assert observation.confidence == 0


def test_feature_observer_rejects_unsafe_topology() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        _observer()
    with pytest.raises(ValueError, match="backend"):
        _observer(
            _template("a", "A", (10, 20), codes={1}, backend=FeatureBackend.ORB),
            _template("b", "B", (30, 40), codes={2}, backend=FeatureBackend.SIFT),
        )
    with pytest.raises(ValueError, match="路线坐标"):
        _observer(
            _template("a-day", "A", (10, 20), codes={1}),
            _template("a-night", "A", (11, 20), codes={1}),
        )


def test_detector_contract_mismatch_raises_before_emitting_route_lock() -> None:
    route_template = FeatureRouteTemplate(
        detector=_Detector(
            label="a",
            waypoint_id="A",
            accepted_codes={1},
            quality=0.8,
            returned_label="stale-a",
        ),
        route_position=(10, 20),
    )
    observer = _observer(route_template)

    with pytest.raises(RuntimeError, match="契约"):
        observer.observe(
            _frame(0, 1),
            scope=ObservationScope(allowed_waypoint_ids=None),
        )


def test_non_boolean_detector_acceptance_cannot_emit_route_lock() -> None:
    route_template = FeatureRouteTemplate(
        detector=_Detector(
            label="a",
            waypoint_id="A",
            accepted_codes={1},
            quality=0.8,
            accepted_value="false",
        ),
        route_position=(10, 20),
    )
    observer = _observer(route_template)

    with pytest.raises(RuntimeError, match="accepted"):
        observer.observe(
            _frame(0, 1),
            scope=ObservationScope(allowed_waypoint_ids=None),
        )


def test_observer_freezes_topology_and_rejects_mutated_detector_contract() -> None:
    detector = _Detector(
        label="a",
        waypoint_id="A",
        accepted_codes={1},
        quality=0.8,
    )
    templates = [FeatureRouteTemplate(detector=detector, route_position=(10, 20))]
    observer = FeatureWaypointObserver(
        templates=templates,
        expected_frame_size=(80, 60),
    )
    templates.append(_template("c", "C", (50, 50), codes={1}))
    detector.waypoint_id = "C"

    with pytest.raises(RuntimeError, match="契约"):
        observer.observe(
            _frame(0, 1),
            scope=ObservationScope(allowed_waypoint_ids=None),
        )
