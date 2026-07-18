import numpy as np
import pytest

from delta_vision.actuator import DryRunActuator
from delta_vision.config import CaptureRegion
from delta_vision.frames import CapturedFrame
from delta_vision.navigation import (
    NavigationPolicy,
    NavigationStatus,
    ObservationScope,
    RouteAction,
    VisualNavigationController,
    WaypointObservation,
    WaypointObserver,
)
from delta_vision.perception import ColorAnchorDetector
from delta_vision.planner import RouteEdge, RouteNode
from delta_vision.template_matching import (
    MatchDecisionPolicy,
    RouteTemplate,
    TemplateAnchorDetector,
    TemplateWaypointObserver,
)


def _graph() -> dict[str, RouteNode]:
    return {
        "A": RouteNode(10, 80, (RouteEdge("B", 1), RouteEdge("C", 5))),
        "B": RouteNode(10, 10, (RouteEdge("D", 1),)),
        "C": RouteNode(80, 80, (RouteEdge("D", 5),)),
        "D": RouteNode(80, 10, ()),
    }


def _frame(sequence: int, x: int | None, y: int | None) -> CapturedFrame:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    if x is not None and y is not None:
        image[y - 2 : y + 3, x - 2 : x + 3] = (0, 255, 0)
    image.setflags(write=False)
    return CapturedFrame(sequence, 1_000_000 + sequence, image, "fixture")


def _observer(*, radius: float = 6) -> WaypointObserver:
    detector = ColorAnchorDetector(
        label="player",
        bgr=(0, 255, 0),
        tolerance=0,
        minimum_area=25,
        confidence_threshold=1,
    )
    return WaypointObserver(
        detector=detector,
        waypoint_positions={node_id: (node.x, node.y) for node_id, node in _graph().items()},
        localization_radius=radius,
    )


def _controller(*, observer=None, **policy_overrides):
    policy_values = {
        "edge_actions": {
            ("A", "B"): "w",
            ("A", "C"): "d",
            ("B", "D"): "d",
            ("C", "D"): "w",
        },
        "pulse_ms": 100,
        "min_progress_px": 3,
        "stuck_after_ms": 300,
        "localization_timeout_ms": 200,
        "max_recovery_attempts": 2,
        "recovery_keys": ("s", "a"),
        "arrival_confirmations": 2,
        "initial_waypoint_confirmations": 3,
        "waypoint_advance_confirmations": 2,
        "relocalization_confirmations": 3,
    }
    policy_values.update(policy_overrides)
    actuator = DryRunActuator(allowed_keys={"w", "a", "s", "d"}, max_key_hold_ms=150)
    controller = VisualNavigationController(
        graph=_graph(),
        observer=observer or _observer(),
        actuator=actuator,
        goal_node_id="D",
        policy=NavigationPolicy(**policy_values),
    )
    return controller, actuator


def _confirm_start(
    controller: VisualNavigationController,
    *,
    sequence: int = 0,
    now_ns: int = 0,
):
    snapshot = None
    for offset in range(3):
        snapshot = controller.on_frame(
            _frame(sequence + offset, 10, 80),
            now_ns=now_ns,
        )
    assert snapshot is not None
    return snapshot


def test_waypoint_observer_localizes_only_unique_nearby_anchor() -> None:
    observer = _observer()
    global_scope = ObservationScope(allowed_waypoint_ids=None)

    accepted = observer.observe(_frame(0, 10, 80), scope=global_scope)
    between = WaypointObserver(
        detector=_observer().detector,
        waypoint_positions={"left": (40, 50), "right": (60, 50)},
        localization_radius=11,
    ).observe(_frame(1, 50, 50), scope=global_scope)
    missing = observer.observe(_frame(2, None, None), scope=global_scope)

    assert accepted.waypoint_id == "A"
    assert accepted.centroid == (10.0, 80.0)
    assert between.waypoint_id is None
    assert between.centroid == (50.0, 50.0)
    assert missing.waypoint_id is None
    assert missing.centroid is None


def test_waypoint_observer_rejects_invalid_topology() -> None:
    with pytest.raises(ValueError, match="坐标"):
        WaypointObserver(
            detector=_observer().detector,
            waypoint_positions={},
            localization_radius=5,
        )
    with pytest.raises(ValueError, match="定位半径"):
        WaypointObserver(
            detector=_observer().detector,
            waypoint_positions={"A": (1, 1)},
            localization_radius=0,
        )


def test_observation_scope_is_immutable_and_empty_scope_fails_closed() -> None:
    with pytest.raises(ValueError, match="frozenset"):
        ObservationScope(allowed_waypoint_ids={"A"})
    with pytest.raises(ValueError, match="节点 ID"):
        ObservationScope(allowed_waypoint_ids=frozenset({""}))

    observation = _observer().observe(
        _frame(0, 10, 80),
        scope=ObservationScope(allowed_waypoint_ids=frozenset()),
    )

    assert observation.centroid is None
    assert observation.waypoint_id is None


def test_initial_localization_requires_three_consecutive_waypoint_frames() -> None:
    controller, actuator = _controller()

    first = controller.on_frame(_frame(0, 10, 80), now_ns=0)
    second = controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)
    confirmed = controller.on_frame(_frame(2, 10, 80), now_ns=20_000_000)

    assert first.status is NavigationStatus.LOCALIZING
    assert first.pending_waypoint_id == "A"
    assert first.confirmation_count == 1
    assert first.confirmation_required == 3
    assert second.status is NavigationStatus.LOCALIZING
    assert second.confirmation_count == 2
    assert confirmed.status is NavigationStatus.NAVIGATING
    assert confirmed.confirmation_count == 0
    assert confirmed.active_key == "w"
    assert [event.kind for event in actuator.events] == ["key_down"]


def test_initial_localization_mismatch_resets_confirmation_streak() -> None:
    controller, actuator = _controller()

    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)
    changed = controller.on_frame(_frame(2, 10, 10), now_ns=20_000_000)
    controller.on_frame(_frame(3, 10, 80), now_ns=30_000_000)
    waiting = controller.on_frame(_frame(4, 10, 80), now_ns=40_000_000)
    confirmed = controller.on_frame(_frame(5, 10, 80), now_ns=50_000_000)

    assert changed.pending_waypoint_id == "B"
    assert changed.confirmation_count == 1
    assert waiting.status is NavigationStatus.LOCALIZING
    assert waiting.confirmation_count == 2
    assert confirmed.status is NavigationStatus.NAVIGATING
    assert [event.kind for event in actuator.events] == ["key_down"]


def test_waypoint_advance_releases_and_requires_two_consecutive_frames() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)
    controller.on_frame(_frame(2, 10, 80), now_ns=20_000_000)

    first = controller.on_frame(_frame(3, 10, 10), now_ns=30_000_000)
    confirmed = controller.on_frame(_frame(4, 10, 10), now_ns=40_000_000)

    assert first.current_node_id == "A"
    assert first.next_node_id == "B"
    assert first.active_key is None
    assert first.pending_waypoint_id == "B"
    assert first.confirmation_count == 1
    assert first.confirmation_required == 2
    assert confirmed.current_node_id == "B"
    assert confirmed.next_node_id == "D"
    assert confirmed.active_key == "d"
    assert [(event.kind, event.key) for event in actuator.events] == [
        ("key_down", "w"),
        ("key_up", "w"),
        ("key_down", "d"),
    ]


def test_missing_anchor_enters_relocalizing_and_requires_three_frames() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)
    controller.on_frame(_frame(2, 10, 80), now_ns=20_000_000)

    missing = controller.on_frame(_frame(3, None, None), now_ns=30_000_000)
    first = controller.on_frame(_frame(4, 10, 80), now_ns=40_000_000)
    second = controller.on_frame(_frame(5, 10, 80), now_ns=50_000_000)
    resumed = controller.on_frame(_frame(6, 10, 80), now_ns=60_000_000)

    assert missing.status is NavigationStatus.RELOCALIZING
    assert first.status is NavigationStatus.RELOCALIZING
    assert first.confirmation_count == 1
    assert second.status is NavigationStatus.RELOCALIZING
    assert second.confirmation_count == 2
    assert resumed.status is NavigationStatus.NAVIGATING
    assert resumed.active_key == "w"
    assert [(event.kind, event.key) for event in actuator.events] == [
        ("key_down", "w"),
        ("key_up", "w"),
        ("key_down", "w"),
    ]


def test_relocalization_at_next_waypoint_advances_after_three_frames() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)
    controller.on_frame(_frame(2, 10, 80), now_ns=20_000_000)
    controller.on_frame(_frame(3, None, None), now_ns=30_000_000)

    controller.on_frame(_frame(4, 10, 10), now_ns=40_000_000)
    controller.on_frame(_frame(5, 10, 10), now_ns=50_000_000)
    resumed = controller.on_frame(_frame(6, 10, 10), now_ns=60_000_000)

    assert resumed.status is NavigationStatus.NAVIGATING
    assert resumed.current_node_id == "B"
    assert resumed.next_node_id == "D"
    assert resumed.active_key == "d"
    assert [(event.kind, event.key) for event in actuator.events] == [
        ("key_down", "w"),
        ("key_up", "w"),
        ("key_down", "d"),
    ]


def test_initial_uncertain_frame_clears_pending_confirmation() -> None:
    controller, actuator = _controller()

    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)
    uncertain = controller.on_frame(_frame(2, None, None), now_ns=20_000_000)
    controller.on_frame(_frame(3, 10, 80), now_ns=30_000_000)
    waiting = controller.on_frame(_frame(4, 10, 80), now_ns=40_000_000)
    confirmed = controller.on_frame(_frame(5, 10, 80), now_ns=50_000_000)

    assert uncertain.confirmation_count == 0
    assert waiting.status is NavigationStatus.LOCALIZING
    assert waiting.confirmation_count == 2
    assert confirmed.status is NavigationStatus.NAVIGATING
    assert [event.kind for event in actuator.events] == ["key_down"]


def test_next_candidate_interruption_resets_advance_confirmation() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)

    first_b = controller.on_frame(_frame(3, 10, 10), now_ns=30_000_000)
    back_at_a = controller.on_frame(_frame(4, 10, 80), now_ns=40_000_000)
    second_first_b = controller.on_frame(_frame(5, 10, 10), now_ns=50_000_000)
    confirmed = controller.on_frame(_frame(6, 10, 10), now_ns=60_000_000)

    assert first_b.confirmation_count == 1
    assert back_at_a.confirmation_count == 0
    assert back_at_a.current_node_id == "A"
    assert second_first_b.current_node_id == "A"
    assert second_first_b.confirmation_count == 1
    assert confirmed.current_node_id == "B"
    assert actuator.pressed_keys == frozenset({"d"})


def test_partial_initial_confirmation_does_not_extend_timeout() -> None:
    controller, actuator = _controller(localization_timeout_ms=100)

    first = controller.on_frame(_frame(0, 10, 80), now_ns=0)
    second = controller.on_frame(_frame(1, 10, 80), now_ns=99_999_999)
    stopped = controller.on_frame(_frame(2, 10, 80), now_ns=100_000_000)

    assert first.status is NavigationStatus.LOCALIZING
    assert second.status is NavigationStatus.LOCALIZING
    assert stopped.status is NavigationStatus.STOPPED
    assert "超时" in (stopped.reason or "")
    assert actuator.events == ()


def test_non_adjacent_jump_is_terminal_while_relocalizing() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)
    controller.on_frame(_frame(3, None, None), now_ns=30_000_000)

    stopped = controller.on_frame(_frame(4, 80, 80), now_ns=40_000_000)

    assert stopped.status is NavigationStatus.STOPPED
    assert stopped.confirmation_count == 0
    assert "观察范围外" in (stopped.reason or "")
    assert actuator.pressed_keys == frozenset()


def test_cached_observer_result_cannot_count_as_fresh_confirmation() -> None:
    class CachedObserver:
        def observe(self, frame, *, scope):
            return WaypointObservation(
                frame_sequence=0,
                captured_at_ns=1_000_000,
                confidence=1,
                centroid=(10, 80),
                waypoint_id="A",
            )

    controller, actuator = _controller(observer=CachedObserver())

    first = controller.on_frame(_frame(0, 10, 80), now_ns=0)
    stopped = controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)

    assert first.status is NavigationStatus.LOCALIZING
    assert stopped.status is NavigationStatus.STOPPED
    assert "观测" in (stopped.reason or "")
    assert actuator.events == ()


def test_zero_capture_timestamp_must_still_be_strictly_increasing() -> None:
    controller, actuator = _controller()
    image = _frame(0, 10, 80).image

    first = controller.on_frame(CapturedFrame(0, 0, image, "fixture"), now_ns=0)
    stopped = controller.on_frame(
        CapturedFrame(1, 0, image, "fixture"),
        now_ns=10_000_000,
    )

    assert first.status is NavigationStatus.LOCALIZING
    assert stopped.status is NavigationStatus.STOPPED
    assert "过期" in (stopped.reason or "")
    assert actuator.events == ()


def test_control_clock_rollback_stops_before_confirmation_can_move() -> None:
    controller, actuator = _controller(localization_timeout_ms=100)

    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 80), now_ns=99_999_999)
    stopped = controller.on_frame(_frame(2, 10, 80), now_ns=10_000_000)

    assert stopped.status is NavigationStatus.STOPPED
    assert "时钟" in (stopped.reason or "")
    assert actuator.events == ()


def test_non_integer_control_clock_stops_without_input() -> None:
    controller, actuator = _controller()

    stopped = controller.on_frame(_frame(0, 10, 80), now_ns=True)

    assert stopped.status is NavigationStatus.STOPPED
    assert "时钟" in (stopped.reason or "")
    assert actuator.events == ()


def test_stale_frame_releases_active_navigation_key() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)

    stopped = controller.on_frame(_frame(2, 10, 80), now_ns=10_000_000)

    assert stopped.status is NavigationStatus.STOPPED
    assert "过期" in (stopped.reason or "")
    assert [(event.kind, event.key) for event in actuator.events] == [
        ("key_down", "w"),
        ("key_up", "w"),
    ]


def test_partial_relocalization_cannot_cross_timeout_boundary() -> None:
    controller, actuator = _controller(localization_timeout_ms=100)
    _confirm_start(controller)
    controller.on_frame(_frame(3, None, None), now_ns=10_000_000)
    controller.on_frame(_frame(4, 10, 80), now_ns=50_000_000)
    controller.on_frame(_frame(5, 10, 80), now_ns=100_000_000)

    stopped = controller.on_frame(_frame(6, 10, 80), now_ns=110_000_000)

    assert stopped.status is NavigationStatus.STOPPED
    assert "超时" in (stopped.reason or "")
    assert actuator.pressed_keys == frozenset()
    assert [(event.kind, event.key) for event in actuator.events] == [
        ("key_down", "w"),
        ("key_up", "w"),
    ]


@pytest.mark.parametrize(
    ("sequence", "captured_at_ns"),
    [
        (None, None),
        (True, 0),
        (0, True),
        (1.5, 0),
        (0, -1),
    ],
)
def test_invalid_frame_metadata_cannot_count_as_confirmation(
    sequence,
    captured_at_ns,
) -> None:
    controller, actuator = _controller()
    image = _frame(0, 10, 80).image
    invalid = CapturedFrame(sequence, captured_at_ns, image, "fixture")

    stopped = controller.on_frame(invalid, now_ns=0)

    assert stopped.status is NavigationStatus.STOPPED
    assert "帧元数据" in (stopped.reason or "")
    assert actuator.events == ()


def test_invalid_frame_metadata_releases_active_key() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)
    image = _frame(3, 10, 80).image

    stopped = controller.on_frame(
        CapturedFrame(None, None, image, "fixture"),
        now_ns=10_000_000,
    )

    assert stopped.status is NavigationStatus.STOPPED
    assert actuator.pressed_keys == frozenset()
    assert [(event.kind, event.key) for event in actuator.events] == [
        ("key_down", "w"),
        ("key_up", "w"),
    ]


def test_boolean_observation_sequence_is_not_equal_to_integer_frame_sequence() -> None:
    class BooleanSequenceObserver:
        def observe(self, frame, *, scope):
            return WaypointObservation(
                frame_sequence=True,
                captured_at_ns=frame.captured_at_ns,
                confidence=1,
                centroid=(10, 80),
                waypoint_id="A",
            )

    controller, actuator = _controller(observer=BooleanSequenceObserver())

    stopped = controller.on_frame(_frame(1, 10, 80), now_ns=0)

    assert stopped.status is NavigationStatus.STOPPED
    assert "观测元数据" in (stopped.reason or "")
    assert actuator.events == ()


def test_controller_uses_weighted_astar_edge_action() -> None:
    controller, actuator = _controller()

    snapshot = _confirm_start(controller)

    assert snapshot.status is NavigationStatus.NAVIGATING
    assert snapshot.route == ("A", "B", "D")
    assert snapshot.current_node_id == "A"
    assert snapshot.next_node_id == "B"
    assert snapshot.active_key == "w"
    assert actuator.pressed_keys == frozenset({"w"})


def test_pulse_releases_at_deadline_without_repeating_key_down() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)

    controller.on_timer(now_ns=99_999_999)
    assert [event.kind for event in actuator.events] == ["key_down"]
    controller.on_timer(now_ns=100_000_000)

    assert [event.kind for event in actuator.events] == ["key_down", "key_up"]
    assert actuator.events[-1].at_ns == 100_000_000
    assert actuator.pressed_keys == frozenset()


def test_new_frame_during_active_pulse_does_not_repeat_key_down() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)

    snapshot = controller.on_frame(_frame(3, 10, 70), now_ns=50_000_000)

    assert snapshot.active_key == "w"
    assert [event.kind for event in actuator.events] == ["key_down"]


def test_route_action_turns_mouse_once_then_repeats_only_key_pulse() -> None:
    controller, actuator = _controller(
        edge_actions={
            ("A", "B"): RouteAction(key="w", mouse_dx=320, mouse_dy=-12),
            ("B", "D"): RouteAction(key="w"),
        }
    )

    _confirm_start(controller)
    controller.on_frame(_frame(3, 10, 70), now_ns=50_000_000)
    controller.on_timer(now_ns=100_000_000)
    controller.on_frame(_frame(4, 10, 60), now_ns=110_000_000)

    assert [event.kind for event in actuator.events] == [
        "mouse_move",
        "key_down",
        "key_up",
        "key_down",
    ]
    mouse_event = actuator.events[0]
    assert (mouse_event.dx, mouse_event.dy) == (320, -12)


def test_route_action_turns_again_only_after_visual_edge_advance() -> None:
    controller, actuator = _controller(
        edge_actions={
            ("A", "B"): RouteAction(key="w", mouse_dx=-80),
            ("B", "D"): RouteAction(key="w", mouse_dx=240),
        }
    )

    _confirm_start(controller)
    controller.on_frame(_frame(3, 10, 10), now_ns=100_000_000)
    controller.on_frame(_frame(4, 10, 10), now_ns=100_000_001)

    mouse_events = [event for event in actuator.events if event.kind == "mouse_move"]
    assert [(event.dx, event.dy) for event in mouse_events] == [(-80, 0), (240, 0)]
    assert actuator.pressed_keys == frozenset({"w"})


def test_same_edge_relocalization_does_not_repeat_mouse_turn() -> None:
    controller, actuator = _controller(
        edge_actions={
            ("A", "B"): RouteAction(key="w", mouse_dx=160),
            ("B", "D"): RouteAction(key="w"),
        }
    )
    _confirm_start(controller)
    controller.on_frame(_frame(3, None, None), now_ns=50_000_000)

    controller.on_frame(_frame(4, 10, 80), now_ns=100_000_000)
    controller.on_frame(_frame(5, 10, 80), now_ns=110_000_000)
    relocalized = controller.on_frame(_frame(6, 10, 80), now_ns=120_000_000)

    assert relocalized.status is NavigationStatus.NAVIGATING
    assert [event.kind for event in actuator.events].count("mouse_move") == 1
    assert actuator.pressed_keys == frozenset({"w"})


def test_low_confidence_releases_active_pulse_and_times_out_without_input() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)

    uncertain = controller.on_frame(_frame(3, None, None), now_ns=50_000_000)
    event_count = len(actuator.events)
    stopped = controller.on_frame(_frame(4, None, None), now_ns=250_000_000)
    controller.on_frame(_frame(5, 10, 80), now_ns=300_000_000)

    assert uncertain.status is NavigationStatus.RELOCALIZING
    assert stopped.status is NavigationStatus.STOPPED
    assert actuator.events[1].kind == "key_up"
    assert len(actuator.events) == event_count
    assert actuator.pressed_keys == frozenset()


def test_stuck_comes_from_visual_non_progress_and_recovery_is_bounded() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)
    controller.on_timer(now_ns=100_000_000)
    controller.on_frame(_frame(3, 10, 60), now_ns=100_000_001)
    controller.on_timer(now_ns=200_000_001)
    controller.on_frame(_frame(4, 10, 60), now_ns=250_000_000)

    recovering = controller.on_frame(_frame(5, 10, 60), now_ns=400_000_001)

    assert recovering.status is NavigationStatus.RECOVERING
    assert recovering.active_key == "s"
    assert [event.kind for event in actuator.events[-2:]] == ["key_up", "key_down"]
    assert actuator.events[-2].key == "w"
    assert actuator.events[-1].key == "s"

    controller.on_timer(now_ns=500_000_001)
    waiting = controller.on_timer(now_ns=550_000_001)
    controller.on_frame(_frame(6, 10, 60), now_ns=550_000_002)
    controller.on_timer(now_ns=650_000_002)
    exhausted = controller.on_frame(_frame(7, 10, 60), now_ns=650_000_003)

    assert waiting.recovery_attempts == 1
    assert exhausted.status is NavigationStatus.STOPPED
    assert exhausted.recovery_attempts == 2
    assert actuator.pressed_keys == frozenset()
    assert [event.key for event in actuator.events if event.kind == "key_down"][-2:] == [
        "s",
        "a",
    ]


def test_visual_progress_after_recovery_resumes_navigation() -> None:
    controller, actuator = _controller(stuck_after_ms=100)
    _confirm_start(controller)
    controller.on_timer(now_ns=100_000_000)
    controller.on_frame(_frame(3, 10, 80), now_ns=100_000_001)

    resumed = controller.on_frame(_frame(4, 10, 60), now_ns=120_000_000)

    assert resumed.status is NavigationStatus.NAVIGATING
    assert resumed.active_key == "w"
    assert resumed.recovery_attempts == 0
    assert actuator.pressed_keys == frozenset({"w"})


def test_recovery_releases_all_keys_and_does_not_repeat_edge_turn() -> None:
    controller, actuator = _controller(
        stuck_after_ms=100,
        edge_actions={
            ("A", "B"): RouteAction(key="w", mouse_dx=160),
            ("B", "D"): RouteAction(key="w"),
        },
    )
    _confirm_start(controller)
    actuator.key_down("d", now_ns=1)
    controller.on_timer(now_ns=100_000_000)

    recovering = controller.on_frame(_frame(3, 10, 80), now_ns=100_000_001)

    assert recovering.status is NavigationStatus.RECOVERING
    assert actuator.pressed_keys == frozenset({"s"})
    assert [event.kind for event in actuator.events].count("mouse_move") == 1
    assert [(event.kind, event.key) for event in actuator.events[-2:]] == [
        ("key_up", "d"),
        ("key_down", "s"),
    ]


def test_route_advances_only_after_visual_waypoint_confirmation() -> None:
    controller, _ = _controller()
    _confirm_start(controller)
    controller.on_timer(now_ns=100_000_000)

    between = controller.on_frame(_frame(3, 10, 40), now_ns=110_000_000)
    first_b = controller.on_frame(_frame(4, 10, 10), now_ns=220_000_000)
    at_b = controller.on_frame(_frame(5, 10, 10), now_ns=230_000_000)

    assert between.current_node_id == "A"
    assert between.next_node_id == "B"
    assert first_b.current_node_id == "A"
    assert first_b.active_key is None
    assert at_b.current_node_id == "B"
    assert at_b.next_node_id == "D"
    assert at_b.active_key == "d"


def test_goal_requires_consecutive_visual_confirmations_and_releases() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)
    controller.on_frame(_frame(3, 10, 10), now_ns=100_000_000)
    controller.on_frame(_frame(4, 10, 10), now_ns=110_000_000)

    controller.on_frame(_frame(5, 80, 10), now_ns=200_000_000)
    first = controller.on_frame(_frame(6, 80, 10), now_ns=210_000_000)
    arrived = controller.on_frame(_frame(7, 80, 10), now_ns=250_000_000)
    event_count = len(actuator.events)
    again = controller.on_frame(_frame(8, 80, 10), now_ns=300_000_000)

    assert first.status is NavigationStatus.NAVIGATING
    assert arrived.status is NavigationStatus.ARRIVED
    assert again.status is NavigationStatus.ARRIVED
    assert actuator.pressed_keys == frozenset()
    assert len(actuator.events) == event_count


def test_unmapped_high_confidence_frame_after_goal_confirmation_stops_safely() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)
    controller.on_frame(_frame(3, 10, 10), now_ns=100_000_000)
    controller.on_frame(_frame(4, 10, 10), now_ns=110_000_000)
    controller.on_frame(_frame(5, 80, 10), now_ns=200_000_000)
    first = controller.on_frame(_frame(6, 80, 10), now_ns=210_000_000)

    unmapped = controller.on_frame(_frame(7, 50, 50), now_ns=250_000_000)

    assert first.status is NavigationStatus.NAVIGATING
    assert unmapped.status is NavigationStatus.STOPPED
    assert unmapped.current_node_id == "D"
    assert unmapped.next_node_id is None
    assert actuator.pressed_keys == frozenset()


def test_unmapped_high_confidence_frame_outside_active_edge_stops_and_releases() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)

    snapshot = controller.on_frame(_frame(3, 50, 50), now_ns=50_000_000)

    assert snapshot.status is NavigationStatus.STOPPED
    assert "观察范围外" in (snapshot.reason or "")
    assert actuator.pressed_keys == frozenset()
    assert [event.kind for event in actuator.events] == ["key_down", "key_up"]


def test_stale_frame_and_manual_stop_are_terminal_and_release_keys() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(1, 10, 80), now_ns=0)

    stale = controller.on_frame(_frame(1, 10, 70), now_ns=1)
    event_count = len(actuator.events)
    controller.on_timer(now_ns=2)
    controller.stop(now_ns=3, reason="人工停止")

    assert stale.status is NavigationStatus.STOPPED
    assert "过期" in (stale.reason or "")
    assert actuator.pressed_keys == frozenset()
    assert len(actuator.events) == event_count


def test_missing_edge_action_stops_before_sending_input() -> None:
    controller, actuator = _controller(edge_actions={})

    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 80), now_ns=10_000_000)
    snapshot = controller.on_frame(_frame(2, 10, 80), now_ns=20_000_000)

    assert snapshot.status is NavigationStatus.STOPPED
    assert "动作" in (snapshot.reason or "")
    assert actuator.events == ()


def test_ambiguous_initial_position_times_out_without_input() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=150)
    observer = WaypointObserver(
        detector=_observer().detector,
        waypoint_positions={"left": (40, 50), "right": (60, 50)},
        localization_radius=11,
    )
    graph = {
        "left": RouteNode(40, 50, (RouteEdge("right", 1),)),
        "right": RouteNode(60, 50, ()),
    }
    policy = NavigationPolicy(
        edge_actions={("left", "right"): "w"},
        pulse_ms=100,
        min_progress_px=2,
        stuck_after_ms=300,
        localization_timeout_ms=100,
        max_recovery_attempts=0,
        recovery_keys=(),
        arrival_confirmations=2,
        initial_waypoint_confirmations=3,
        waypoint_advance_confirmations=2,
        relocalization_confirmations=3,
    )
    controller = VisualNavigationController(
        graph=graph,
        observer=observer,
        actuator=actuator,
        goal_node_id="right",
        policy=policy,
    )

    first = controller.on_frame(_frame(0, 50, 50), now_ns=0)
    stopped = controller.on_frame(_frame(1, 50, 50), now_ns=100_000_000)

    assert first.status is NavigationStatus.LOCALIZING
    assert stopped.status is NavigationStatus.STOPPED
    assert actuator.events == ()


def test_localizing_times_out_even_when_screenshot_stream_stops() -> None:
    controller, actuator = _controller()

    first = controller.on_frame(_frame(0, None, None), now_ns=0)
    stopped = controller.on_timer(now_ns=200_000_000)

    assert first.status is NavigationStatus.LOCALIZING
    assert stopped.status is NavigationStatus.STOPPED
    assert actuator.events == ()


def test_non_adjacent_visual_jump_stops_and_releases() -> None:
    controller, actuator = _controller()
    _confirm_start(controller)

    snapshot = controller.on_frame(_frame(3, 80, 80), now_ns=50_000_000)

    assert snapshot.status is NavigationStatus.STOPPED
    assert "非相邻" in (snapshot.reason or "")
    assert actuator.pressed_keys == frozenset()


def test_template_navigation_uses_current_and_next_waypoints_as_match_candidates() -> None:
    current_template = np.random.default_rng(20260716).integers(
        0, 256, size=(12, 16, 3), dtype=np.uint8
    )
    distant_template = np.array(current_template, copy=True)
    distant_template[0, 0] = 255 - distant_template[0, 0]
    next_template = np.random.default_rng(99).integers(
        0, 256, size=current_template.shape, dtype=np.uint8
    )
    off_route_template = np.random.default_rng(1234).integers(
        0, 256, size=current_template.shape, dtype=np.uint8
    )

    def detector(template: np.ndarray) -> TemplateAnchorDetector:
        return TemplateAnchorDetector(
            label="route",
            template=template,
            search_roi=CaptureRegion(0, 0, 100, 100),
            scales=(1.0,),
            policy=MatchDecisionPolicy(score_threshold=0.8, minimum_margin=0.05),
            nms_radius_px=18,
        )

    observer = TemplateWaypointObserver(
        templates=(
            RouteTemplate("current", detector(current_template), (10, 80), "A"),
            RouteTemplate("next", detector(next_template), (10, 10), "B"),
            RouteTemplate("distant", detector(distant_template), (80, 10), "D"),
            RouteTemplate("off-route", detector(off_route_template), (80, 80), "C"),
            RouteTemplate(
                "off-route-second-view",
                detector(off_route_template),
                (80, 80),
                "C",
            ),
        ),
        expected_frame_size=(100, 100),
        minimum_template_margin=0,
    )
    actuator = DryRunActuator(allowed_keys={"w", "d"}, max_key_hold_ms=150)
    controller = VisualNavigationController(
        graph={
            "A": RouteNode(10, 80, (RouteEdge("B", 1),)),
            "B": RouteNode(10, 10, (RouteEdge("D", 1),)),
            "C": RouteNode(80, 80, (RouteEdge("D", 1),)),
            "D": RouteNode(80, 10, ()),
        },
        observer=observer,
        actuator=actuator,
        goal_node_id="D",
        policy=NavigationPolicy(
            edge_actions={
                ("A", "B"): "w",
                ("B", "D"): "d",
            },
            pulse_ms=100,
            min_progress_px=3,
            stuck_after_ms=300,
            localization_timeout_ms=200,
            max_recovery_attempts=0,
            recovery_keys=(),
            arrival_confirmations=2,
            initial_waypoint_confirmations=3,
            waypoint_advance_confirmations=2,
            relocalization_confirmations=3,
        ),
    )

    first_image = np.zeros((100, 100, 3), dtype=np.uint8)
    first_image[40:52, 42:58] = current_template
    first_image.setflags(write=False)
    controller.on_frame(CapturedFrame(0, 1, first_image, "fixture"), now_ns=0)
    controller.on_frame(CapturedFrame(1, 2, first_image, "fixture"), now_ns=1)
    controller.on_frame(CapturedFrame(2, 3, first_image, "fixture"), now_ns=2)

    missing_image = np.zeros((100, 100, 3), dtype=np.uint8)
    missing_image.setflags(write=False)
    localizing = controller.on_frame(
        CapturedFrame(3, 4, missing_image, "fixture"),
        now_ns=50_000_000,
    )

    second_image = np.zeros((100, 100, 3), dtype=np.uint8)
    second_image[40:52, 42:58] = distant_template
    second_image.setflags(write=False)
    controller.on_frame(
        CapturedFrame(4, 5, second_image, "fixture"),
        now_ns=100_000_000,
    )
    controller.on_frame(
        CapturedFrame(5, 6, second_image, "fixture"),
        now_ns=110_000_000,
    )
    snapshot = controller.on_frame(
        CapturedFrame(6, 7, second_image, "fixture"),
        now_ns=120_000_000,
    )

    assert localizing.status is NavigationStatus.RELOCALIZING
    assert snapshot.status is NavigationStatus.NAVIGATING
    assert snapshot.current_node_id == "A"
    assert snapshot.next_node_id == "B"
    assert actuator.pressed_keys == frozenset({"w"})

    off_route_image = np.zeros((100, 100, 3), dtype=np.uint8)
    off_route_image[40:52, 42:58] = off_route_template
    off_route_image.setflags(write=False)
    stopped = controller.on_frame(
        CapturedFrame(7, 8, off_route_image, "fixture"),
        now_ns=150_000_000,
    )

    assert stopped.status is NavigationStatus.STOPPED
    assert "C" in (stopped.reason or "")
    assert actuator.pressed_keys == frozenset()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pulse_ms", 0),
        ("pulse_ms", float("inf")),
        ("min_progress_px", 0),
        ("stuck_after_ms", 0),
        ("localization_timeout_ms", 0),
        ("max_recovery_attempts", -1),
        ("arrival_confirmations", 0),
        ("initial_waypoint_confirmations", 2),
        ("initial_waypoint_confirmations", True),
        ("waypoint_advance_confirmations", 1),
        ("waypoint_advance_confirmations", True),
        ("relocalization_confirmations", 2),
        ("relocalization_confirmations", True),
    ],
)
def test_navigation_policy_rejects_unsafe_values(field: str, value: int) -> None:
    values = {
        "edge_actions": {("A", "B"): "w"},
        "pulse_ms": 100,
        "min_progress_px": 3,
        "stuck_after_ms": 300,
        "localization_timeout_ms": 200,
        "max_recovery_attempts": 2,
        "recovery_keys": ("s",),
        "arrival_confirmations": 2,
        "initial_waypoint_confirmations": 3,
        "waypoint_advance_confirmations": 2,
        "relocalization_confirmations": 3,
    }
    values[field] = value

    with pytest.raises(ValueError):
        NavigationPolicy(**values)


def test_navigation_policy_requires_recovery_key_when_recovery_is_enabled() -> None:
    with pytest.raises(ValueError, match="恢复按键"):
        NavigationPolicy(
            edge_actions={("A", "B"): "w"},
            pulse_ms=100,
            min_progress_px=3,
            stuck_after_ms=300,
            localization_timeout_ms=200,
            max_recovery_attempts=1,
            recovery_keys=(),
            arrival_confirmations=2,
            initial_waypoint_confirmations=3,
            waypoint_advance_confirmations=2,
            relocalization_confirmations=3,
        )


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"key": ""}, "按键"),
        ({"key": "w", "mouse_dx": True}, "鼠标"),
        ({"key": "w", "mouse_dx": 4097}, "鼠标"),
        ({"key": "w", "mouse_dy": -4097}, "鼠标"),
    ],
)
def test_route_action_rejects_unsafe_contract(kwargs, error_match: str) -> None:
    with pytest.raises(ValueError, match=error_match):
        RouteAction(**kwargs)
