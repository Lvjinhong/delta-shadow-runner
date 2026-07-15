"""固定 waypoint 图的确定性 A* 路线规划。"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteEdge:
    target_node_id: str
    cost: float


@dataclass(frozen=True, slots=True)
class RouteNode:
    x: float
    y: float
    edges: tuple[RouteEdge, ...]


def _validate_graph(graph: dict[str, RouteNode]) -> None:
    for node_id, node in graph.items():
        if not math.isfinite(node.x) or not math.isfinite(node.y):
            raise ValueError(f'节点 "{node_id}" 的坐标必须是有限数')
        for edge in node.edges:
            if edge.target_node_id not in graph:
                raise ValueError(
                    f'节点 "{node_id}" 的边指向未知节点 "{edge.target_node_id}"'
                )
            if not math.isfinite(edge.cost) or edge.cost < 0:
                raise ValueError(
                    f'节点 "{node_id}" 到 "{edge.target_node_id}" 的代价必须是非负有限数'
                )


def _distance(source: RouteNode, target: RouteNode) -> float:
    return math.hypot(target.x - source.x, target.y - source.y)


def _heuristic_scale(graph: dict[str, RouteNode]) -> float:
    scales: list[float] = []
    for node in graph.values():
        for edge in node.edges:
            distance = _distance(node, graph[edge.target_node_id])
            if distance > 0:
                scales.append(edge.cost / distance)
    return min(scales, default=0.0)


def _reconstruct_path(
    came_from: dict[str, str], start_node_id: str, target_node_id: str
) -> tuple[str, ...]:
    path = [target_node_id]
    while path[-1] != start_node_id:
        path.append(came_from[path[-1]])
    path.reverse()
    return tuple(path)


def find_shortest_path(
    graph: dict[str, RouteNode], start_node_id: str, target_node_id: str
) -> tuple[str, ...]:
    _validate_graph(graph)
    if start_node_id not in graph:
        raise ValueError(f'路线图中不存在起点节点 "{start_node_id}"')
    if target_node_id not in graph:
        raise ValueError(f'路线图中不存在终点节点 "{target_node_id}"')
    if start_node_id == target_node_id:
        return (start_node_id,)

    target = graph[target_node_id]
    heuristic_scale = _heuristic_scale(graph)
    came_from: dict[str, str] = {}
    traveled = {start_node_id: 0.0}
    queue = [(heuristic_scale * _distance(graph[start_node_id], target), start_node_id)]

    while queue:
        estimated_total, current_node_id = heapq.heappop(queue)
        known_estimate = traveled[current_node_id] + heuristic_scale * _distance(
            graph[current_node_id], target
        )
        if estimated_total > known_estimate:
            continue
        if current_node_id == target_node_id:
            return _reconstruct_path(came_from, start_node_id, target_node_id)

        current_cost = traveled[current_node_id]
        for edge in graph[current_node_id].edges:
            next_cost = current_cost + edge.cost
            if next_cost >= traveled.get(edge.target_node_id, math.inf):
                continue
            traveled[edge.target_node_id] = next_cost
            came_from[edge.target_node_id] = current_node_id
            estimate = next_cost + heuristic_scale * _distance(
                graph[edge.target_node_id], target
            )
            heapq.heappush(queue, (estimate, edge.target_node_id))

    raise ValueError(f'无法从节点 "{start_node_id}" 到达节点 "{target_node_id}"')
