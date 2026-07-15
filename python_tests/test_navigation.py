import numpy as np
import pytest

from delta_vision.actuator import DryRunActuator
from delta_vision.frames import CapturedFrame
from delta_vision.navigation import (
    NavigationPolicy,
    NavigationStatus,
    RouteAction,
    VisualNavigationController,
    WaypointObserver,
)
from delta_vision.perception import ColorAnchorDetector
from delta_vision.planner import RouteEdge, RouteNode


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
        waypoint_positions={
            node_id: (node.x, node.y) for node_id, node in _graph().items()
        },
        localization_radius=radius,
    )


def _controller(**policy_overrides):
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
    }
    policy_values.update(policy_overrides)
    actuator = DryRunActuator(
        allowed_keys={"w", "a", "s", "d"}, max_key_hold_ms=150
    )
    controller = VisualNavigationController(
        graph=_graph(),
        observer=_observer(),
        actuator=actuator,
        goal_node_id="D",
        policy=NavigationPolicy(**policy_values),
    )
    return controller, actuator


def test_waypoint_observer_localizes_only_unique_nearby_anchor() -> None:
    observer = _observer()

    accepted = observer.observe(_frame(0, 10, 80))
    between = WaypointObserver(
        detector=_observer().detector,
        waypoint_positions={"left": (40, 50), "right": (60, 50)},
        localization_radius=11,
    ).observe(_frame(1, 50, 50))
    missing = observer.observe(_frame(2, None, None))

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


def test_controller_uses_weighted_astar_edge_action() -> None:
    controller, actuator = _controller()

    snapshot = controller.on_frame(_frame(0, 10, 80), now_ns=0)

    assert snapshot.status is NavigationStatus.NAVIGATING
    assert snapshot.route == ("A", "B", "D")
    assert snapshot.current_node_id == "A"
    assert snapshot.next_node_id == "B"
    assert snapshot.active_key == "w"
    assert actuator.pressed_keys == frozenset({"w"})


def test_pulse_releases_at_deadline_without_repeating_key_down() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)

    controller.on_timer(now_ns=99_999_999)
    assert [event.kind for event in actuator.events] == ["key_down"]
    controller.on_timer(now_ns=100_000_000)

    assert [event.kind for event in actuator.events] == ["key_down", "key_up"]
    assert actuator.events[-1].at_ns == 100_000_000
    assert actuator.pressed_keys == frozenset()


def test_new_frame_during_active_pulse_does_not_repeat_key_down() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)

    snapshot = controller.on_frame(_frame(1, 10, 70), now_ns=50_000_000)

    assert snapshot.active_key == "w"
    assert [event.kind for event in actuator.events] == ["key_down"]


def test_route_action_turns_mouse_once_then_repeats_only_key_pulse() -> None:
    controller, actuator = _controller(
        edge_actions={
            ("A", "B"): RouteAction(key="w", mouse_dx=320, mouse_dy=-12),
            ("B", "D"): RouteAction(key="w"),
        }
    )

    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 70), now_ns=50_000_000)
    controller.on_timer(now_ns=100_000_000)
    controller.on_frame(_frame(2, 10, 60), now_ns=110_000_000)

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

    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 10), now_ns=100_000_000)

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
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, None, None), now_ns=50_000_000)

    relocalized = controller.on_frame(_frame(2, 10, 80), now_ns=100_000_000)

    assert relocalized.status is NavigationStatus.NAVIGATING
    assert [event.kind for event in actuator.events].count("mouse_move") == 1
    assert actuator.pressed_keys == frozenset({"w"})


def test_low_confidence_releases_active_pulse_and_times_out_without_input() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)

    uncertain = controller.on_frame(_frame(1, None, None), now_ns=50_000_000)
    event_count = len(actuator.events)
    stopped = controller.on_frame(_frame(2, None, None), now_ns=250_000_000)
    controller.on_frame(_frame(3, 10, 80), now_ns=300_000_000)

    assert uncertain.status is NavigationStatus.LOCALIZING
    assert stopped.status is NavigationStatus.STOPPED
    assert actuator.events[1].kind == "key_up"
    assert len(actuator.events) == event_count
    assert actuator.pressed_keys == frozenset()


def test_stuck_comes_from_visual_non_progress_and_recovery_is_bounded() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_timer(now_ns=100_000_000)
    controller.on_frame(_frame(1, 10, 60), now_ns=100_000_001)
    controller.on_timer(now_ns=200_000_001)
    controller.on_frame(_frame(2, 10, 60), now_ns=250_000_000)

    recovering = controller.on_frame(_frame(3, 10, 60), now_ns=400_000_001)

    assert recovering.status is NavigationStatus.RECOVERING
    assert recovering.active_key == "s"
    assert [event.kind for event in actuator.events[-2:]] == ["key_up", "key_down"]
    assert actuator.events[-2].key == "w"
    assert actuator.events[-1].key == "s"

    controller.on_timer(now_ns=500_000_001)
    waiting = controller.on_timer(now_ns=550_000_001)
    controller.on_frame(_frame(4, 10, 60), now_ns=550_000_002)
    controller.on_timer(now_ns=650_000_002)
    exhausted = controller.on_frame(_frame(5, 10, 60), now_ns=650_000_003)

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
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_timer(now_ns=100_000_000)
    controller.on_frame(_frame(1, 10, 80), now_ns=100_000_001)

    resumed = controller.on_frame(_frame(2, 10, 60), now_ns=120_000_000)

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
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    actuator.key_down("d", now_ns=1)
    controller.on_timer(now_ns=100_000_000)

    recovering = controller.on_frame(_frame(1, 10, 80), now_ns=100_000_001)

    assert recovering.status is NavigationStatus.RECOVERING
    assert actuator.pressed_keys == frozenset({"s"})
    assert [event.kind for event in actuator.events].count("mouse_move") == 1
    assert [(event.kind, event.key) for event in actuator.events[-2:]] == [
        ("key_up", "d"),
        ("key_down", "s"),
    ]


def test_route_advances_only_after_visual_waypoint_confirmation() -> None:
    controller, _ = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_timer(now_ns=100_000_000)

    between = controller.on_frame(_frame(1, 10, 40), now_ns=110_000_000)
    at_b = controller.on_frame(_frame(2, 10, 10), now_ns=220_000_000)

    assert between.current_node_id == "A"
    assert between.next_node_id == "B"
    assert at_b.current_node_id == "B"
    assert at_b.next_node_id == "D"
    assert at_b.active_key == "d"


def test_goal_requires_consecutive_visual_confirmations_and_releases() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 10), now_ns=100_000_000)

    first = controller.on_frame(_frame(2, 80, 10), now_ns=200_000_000)
    arrived = controller.on_frame(_frame(3, 80, 10), now_ns=250_000_000)
    event_count = len(actuator.events)
    again = controller.on_frame(_frame(4, 80, 10), now_ns=300_000_000)

    assert first.status is NavigationStatus.NAVIGATING
    assert arrived.status is NavigationStatus.ARRIVED
    assert again.status is NavigationStatus.ARRIVED
    assert actuator.pressed_keys == frozenset()
    assert len(actuator.events) == event_count


def test_unmapped_high_confidence_frame_after_goal_confirmation_does_not_advance_route() -> None:
    controller, actuator = _controller()
    controller.on_frame(_frame(0, 10, 80), now_ns=0)
    controller.on_frame(_frame(1, 10, 10), now_ns=100_000_000)
    first = controller.on_frame(_frame(2, 80, 10), now_ns=200_000_000)

    unmapped = controller.on_frame(_frame(3, 50, 50), now_ns=250_000_000)

    assert first.status is NavigationStatus.NAVIGATING
    assert unmapped.status is NavigationStatus.NAVIGATING
    assert unmapped.current_node_id == "D"
    assert unmapped.next_node_id is None
    assert actuator.pressed_keys == frozenset()


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

    snapshot = controller.on_frame(_frame(0, 10, 80), now_ns=0)

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
    controller.on_frame(_frame(0, 10, 80), now_ns=0)

    snapshot = controller.on_frame(_frame(1, 80, 80), now_ns=50_000_000)

    assert snapshot.status is NavigationStatus.STOPPED
    assert "非相邻" in (snapshot.reason or "")
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
