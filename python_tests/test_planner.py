import copy
import heapq
import math
import random
from itertools import pairwise

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


def test_find_shortest_path_rejects_unknown_start() -> None:
    with pytest.raises(ValueError, match="不存在起点节点"):
        find_shortest_path(_graph(), "MISSING", "C")


def test_find_shortest_path_rejects_negative_edge_cost() -> None:
    graph = {"A": RouteNode(0, 0, (RouteEdge("B", -1),)), "B": RouteNode(1, 0, ())}

    with pytest.raises(ValueError, match="代价必须是非负有限数"):
        find_shortest_path(graph, "A", "B")


def test_find_shortest_path_reports_unreachable_target() -> None:
    graph = {"A": RouteNode(0, 0, ()), "B": RouteNode(1, 0, ())}

    with pytest.raises(ValueError, match="无法从节点"):
        find_shortest_path(graph, "A", "B")


def test_find_shortest_path_is_not_misled_by_geometrically_near_expensive_route() -> None:
    graph = {
        "start": RouteNode(
            0,
            0,
            (RouteEdge("near", 100), RouteEdge("detour", 1)),
        ),
        "near": RouteNode(9, 0, (RouteEdge("target", 1),)),
        "detour": RouteNode(-1_000, 0, (RouteEdge("target", 1),)),
        "target": RouteNode(10, 0, ()),
    }

    assert find_shortest_path(graph, "start", "target") == (
        "start",
        "detour",
        "target",
    )


def test_find_shortest_path_handles_overflowing_coordinate_distances() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    graph = {
        "start": RouteNode(
            maximum,
            0,
            (RouteEdge("target", 100), RouteEdge("bridge", 1)),
        ),
        "bridge": RouteNode(maximum, 0, (RouteEdge("far", 1),)),
        "far": RouteNode(-maximum, 0, (RouteEdge("target", 1),)),
        "target": RouteNode(maximum / 2, 0, ()),
    }

    assert find_shortest_path(graph, "start", "target") == (
        "start",
        "bridge",
        "far",
        "target",
    )


@pytest.mark.parametrize(
    ("x", "y"),
    [(math.nan, 0), (0, math.inf), (0, -math.inf)],
)
def test_find_shortest_path_rejects_non_finite_coordinates(x: float, y: float) -> None:
    graph = {
        "start": RouteNode(0, 0, (RouteEdge("target", 1),)),
        "target": RouteNode(x, y, ()),
    }

    with pytest.raises(ValueError, match=r"target.*坐标必须是有限数"):
        find_shortest_path(graph, "start", "target")


def test_find_shortest_path_reports_accumulated_cost_overflowing_edge() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    graph = {
        "start": RouteNode(0, 0, (RouteEdge("middle", maximum),)),
        "middle": RouteNode(1, 0, (RouteEdge("target", maximum),)),
        "target": RouteNode(2, 0, ()),
    }

    with pytest.raises(ValueError, match=r"middle.*target.*累计代价"):
        find_shortest_path(graph, "start", "target")


def test_find_shortest_path_accepts_extreme_but_finite_coordinates() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    graph = {
        "start": RouteNode(-maximum, 0, (RouteEdge("target", maximum),)),
        "target": RouteNode(maximum, 0, ()),
    }

    assert find_shortest_path(graph, "start", "target") == ("start", "target")


def test_find_shortest_path_supports_zero_cost_edges() -> None:
    graph = {
        "start": RouteNode(
            0,
            0,
            (RouteEdge("target", 1), RouteEdge("free", 0)),
        ),
        "free": RouteNode(1, 0, (RouteEdge("target", 0),)),
        "target": RouteNode(2, 0, ()),
    }

    assert find_shortest_path(graph, "start", "target") == (
        "start",
        "free",
        "target",
    )


def test_find_shortest_path_uses_lower_cost_duplicate_edge() -> None:
    graph = {
        "start": RouteNode(
            0,
            0,
            (RouteEdge("target", 10), RouteEdge("target", 1)),
        ),
        "target": RouteNode(1, 0, ()),
    }

    assert find_shortest_path(graph, "start", "target") == ("start", "target")


def test_find_shortest_path_rejects_dangling_edge() -> None:
    graph = {
        "start": RouteNode(0, 0, (RouteEdge("missing", 1),)),
        "target": RouteNode(1, 0, ()),
    }

    with pytest.raises(ValueError, match=r"start.*missing"):
        find_shortest_path(graph, "start", "target")


@pytest.mark.parametrize("cost", [-1, math.nan, math.inf, -math.inf])
def test_find_shortest_path_rejects_invalid_edge_cost(cost: float) -> None:
    graph = {
        "start": RouteNode(0, 0, (RouteEdge("target", cost),)),
        "target": RouteNode(1, 0, ()),
    }

    with pytest.raises(ValueError, match=r"start.*target.*代价"):
        find_shortest_path(graph, "start", "target")


def test_find_shortest_path_does_not_mutate_input_graph() -> None:
    graph = _graph()
    before = copy.deepcopy(graph)

    assert find_shortest_path(graph, "A", "C") == ("A", "B", "C")
    assert graph == before


def _path_cost(graph: dict[str, RouteNode], path: tuple[str, ...]) -> float:
    total = 0.0
    for source, target in pairwise(path):
        total += min(
            edge.cost
            for edge in graph[source].edges
            if edge.target_node_id == target
        )
    return total


def _dijkstra_cost(
    graph: dict[str, RouteNode], start_node_id: str, target_node_id: str
) -> float:
    queue = [(0.0, start_node_id)]
    best = {start_node_id: 0.0}
    while queue:
        cost, node_id = heapq.heappop(queue)
        if cost != best[node_id]:
            continue
        if node_id == target_node_id:
            return cost
        for edge in graph[node_id].edges:
            candidate = cost + edge.cost
            if candidate >= best.get(edge.target_node_id, math.inf):
                continue
            best[edge.target_node_id] = candidate
            heapq.heappush(queue, (candidate, edge.target_node_id))
    raise AssertionError("生成器必须保证目标可达")


def test_find_shortest_path_matches_dijkstra_on_1000_seeded_graphs() -> None:
    random_source = random.Random(20260716)
    for _ in range(1_000):
        node_count = random_source.randint(2, 8)
        node_ids = tuple(f"node-{index}" for index in range(node_count))
        coordinates = {
            node_id: (
                random_source.uniform(-1_000, 1_000),
                random_source.uniform(-1_000, 1_000),
            )
            for node_id in node_ids
        }
        edges_by_node: dict[str, list[RouteEdge]] = {
            node_id: [] for node_id in node_ids
        }
        # 先放入一条链，保证每次样本都有可比较的可达路径。
        for index in range(node_count - 1):
            edges_by_node[node_ids[index]].append(
                RouteEdge(node_ids[index + 1], random_source.uniform(0, 100))
            )
        for source in node_ids:
            for target in node_ids:
                if source != target and random_source.random() < 0.25:
                    edges_by_node[source].append(
                        RouteEdge(target, random_source.uniform(0, 100))
                    )
        graph = {
            node_id: RouteNode(
                coordinates[node_id][0],
                coordinates[node_id][1],
                tuple(edges_by_node[node_id]),
            )
            for node_id in node_ids
        }

        path = find_shortest_path(graph, node_ids[0], node_ids[-1])

        assert _path_cost(graph, path) == pytest.approx(
            _dijkstra_cost(graph, node_ids[0], node_ids[-1])
        )
