import pytest

from delta_vision.planner import RouteEdge, RouteNode, find_shortest_path


def _graph() -> dict[str, RouteNode]:
    return {
        "A": RouteNode(x=0, y=0, edges=(RouteEdge("B", 1), RouteEdge("C", 5))),
        "B": RouteNode(x=1, y=0, edges=(RouteEdge("C", 1),)),
        "C": RouteNode(x=2, y=0, edges=()),
    }


def test_find_shortest_path_chooses_lowest_cost_route() -> None:
    assert find_shortest_path(_graph(), "A", "C") == ("A", "B", "C")


def test_find_shortest_path_returns_single_node_for_same_start_and_target() -> None:
    assert find_shortest_path(_graph(), "B", "B") == ("B",)


def test_find_shortest_path_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="不存在终点节点"):
        find_shortest_path(_graph(), "A", "MISSING")


def test_find_shortest_path_rejects_negative_edge_cost() -> None:
    graph = {"A": RouteNode(0, 0, (RouteEdge("B", -1),)), "B": RouteNode(1, 0, ())}

    with pytest.raises(ValueError, match="代价必须是非负有限数"):
        find_shortest_path(graph, "A", "B")


def test_find_shortest_path_reports_unreachable_target() -> None:
    graph = {"A": RouteNode(0, 0, ()), "B": RouteNode(1, 0, ())}

    with pytest.raises(ValueError, match="无法从节点"):
        find_shortest_path(graph, "A", "B")
