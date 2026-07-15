"""只从截图观测驱动的 waypoint 导航状态机。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .frames import CapturedFrame
from .planner import RouteNode, find_shortest_path


class _Actuator(Protocol):
    @property
    def pressed_keys(self) -> frozenset[str]: ...

    def key_down(self, key: str, *, now_ns: int) -> None: ...

    def key_up(self, key: str, *, now_ns: int, reason: str | None = None) -> None: ...

    def move_mouse_relative(self, dx: int, dy: int, *, now_ns: int) -> None: ...

    def release_all(self, *, now_ns: int, reason: str) -> None: ...


class AnchorDetection(Protocol):
    confidence: float
    centroid: tuple[float, float] | None


class AnchorDetector(Protocol):
    def detect(self, image: NDArray[np.uint8]) -> AnchorDetection: ...


class NavigationStatus(StrEnum):
    LOCALIZING = "localizing"
    NAVIGATING = "navigating"
    RECOVERING = "recovering"
    ARRIVED = "arrived"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class WaypointObservation:
    frame_sequence: int
    captured_at_ns: int
    confidence: float
    centroid: tuple[float, float] | None
    waypoint_id: str | None
    out_of_scope_waypoint_id: str | None = None
    scope_violation: bool = False


@dataclass(frozen=True, slots=True)
class ObservationScope:
    """显式声明本帧允许全局定位，或仅在给定路线节点内定位。"""

    allowed_waypoint_ids: frozenset[str] | None

    def __post_init__(self) -> None:
        allowed = self.allowed_waypoint_ids
        if allowed is not None and (
            not isinstance(allowed, frozenset)
            or any(not isinstance(node_id, str) or not node_id for node_id in allowed)
        ):
            raise ValueError("观察范围必须是 null 或仅含有效节点 ID 的 frozenset")


class WaypointObservationSource(Protocol):
    def observe(
        self,
        frame: CapturedFrame,
        *,
        scope: ObservationScope,
    ) -> WaypointObservation: ...


class WaypointObserver:
    """把颜色锚点截图离散化为唯一的附近 waypoint。"""

    def __init__(
        self,
        *,
        detector: AnchorDetector,
        waypoint_positions: Mapping[str, tuple[float, float]],
        localization_radius: float,
    ) -> None:
        if not waypoint_positions:
            raise ValueError("waypoint 坐标不能为空")
        if not math.isfinite(localization_radius) or localization_radius <= 0:
            raise ValueError("定位半径必须是正有限数")
        self._detector = detector
        self._waypoint_positions = MappingProxyType(dict(waypoint_positions))
        self._localization_radius = localization_radius

    @property
    def detector(self) -> AnchorDetector:
        return self._detector

    @staticmethod
    def _distance_to_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        squared_length = delta_x * delta_x + delta_y * delta_y
        if squared_length <= 1e-12:
            return math.dist(point, start)
        projection = (
            (point[0] - start[0]) * delta_x
            + (point[1] - start[1]) * delta_y
        ) / squared_length
        bounded = max(0.0, min(1.0, projection))
        nearest = (start[0] + bounded * delta_x, start[1] + bounded * delta_y)
        return math.dist(point, nearest)

    def _distance_to_scope(
        self,
        centroid: tuple[float, float],
        waypoint_positions: Mapping[str, tuple[float, float]],
    ) -> float:
        points = tuple(waypoint_positions.values())
        if len(points) == 1:
            return math.dist(centroid, points[0])
        return min(
            self._distance_to_segment(centroid, start, end)
            for index, start in enumerate(points)
            for end in points[index + 1 :]
        )

    def observe(
        self,
        frame: CapturedFrame,
        *,
        scope: ObservationScope,
    ) -> WaypointObservation:
        detected = self._detector.detect(frame.image)
        centroid = detected.centroid
        waypoint_id = None
        out_of_scope_waypoint_id = None
        scope_violation = False
        if centroid is not None:
            allowed = scope.allowed_waypoint_ids
            global_distances = sorted(
                (
                    math.hypot(centroid[0] - point[0], centroid[1] - point[1]),
                    node_id,
                )
                for node_id, point in self._waypoint_positions.items()
            )
            global_nearest_distance, global_nearest_id = global_distances[0]
            global_tied = len(global_distances) > 1 and math.isclose(
                global_nearest_distance,
                global_distances[1][0],
                rel_tol=0,
                abs_tol=1e-9,
            )
            global_waypoint_id = (
                global_nearest_id
                if global_nearest_distance <= self._localization_radius
                and not global_tied
                else None
            )
            waypoint_positions = (
                self._waypoint_positions
                if allowed is None
                else {
                    node_id: point
                    for node_id, point in self._waypoint_positions.items()
                    if node_id in allowed
                }
            )
            if not waypoint_positions:
                centroid = None
                scope_violation = True
            elif global_waypoint_id is not None and (
                allowed is not None and global_waypoint_id not in allowed
            ):
                out_of_scope_waypoint_id = global_waypoint_id
                scope_violation = True
            elif allowed is not None and (
                self._distance_to_scope(centroid, waypoint_positions)
                > self._localization_radius
            ):
                scope_violation = True
            else:
                distances = sorted(
                    (
                        math.hypot(centroid[0] - point[0], centroid[1] - point[1]),
                        node_id,
                    )
                    for node_id, point in waypoint_positions.items()
                )
                nearest_distance, nearest_id = distances[0]
                tied = len(distances) > 1 and math.isclose(
                    nearest_distance, distances[1][0], rel_tol=0, abs_tol=1e-9
                )
                if nearest_distance <= self._localization_radius and not tied:
                    waypoint_id = nearest_id
        return WaypointObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            confidence=detected.confidence,
            centroid=centroid,
            waypoint_id=waypoint_id,
            out_of_scope_waypoint_id=out_of_scope_waypoint_id,
            scope_violation=scope_violation,
        )


@dataclass(frozen=True, slots=True)
class RouteAction:
    """进入路线边时转向一次，并重复发送有界移动脉冲。"""

    key: str
    mouse_dx: int = 0
    mouse_dy: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("路线动作按键必须是非空字符串")
        if any(
            type(delta) is not int or abs(delta) > 4096
            for delta in (self.mouse_dx, self.mouse_dy)
        ):
            raise ValueError("相对鼠标位移必须是 -4096..4096 的整数")


@dataclass(frozen=True, slots=True)
class NavigationPolicy:
    edge_actions: Mapping[tuple[str, str], RouteAction]
    pulse_ms: int
    min_progress_px: float
    stuck_after_ms: int
    localization_timeout_ms: int
    max_recovery_attempts: int
    recovery_keys: tuple[str, ...]
    arrival_confirmations: int

    def __post_init__(self) -> None:
        if type(self.pulse_ms) is not int or self.pulse_ms <= 0:
            raise ValueError("动作脉冲时长必须为正数")
        if not math.isfinite(self.min_progress_px) or self.min_progress_px <= 0:
            raise ValueError("最小视觉进展必须是正有限数")
        if isinstance(self.stuck_after_ms, bool) or self.stuck_after_ms <= 0:
            raise ValueError("卡住超时必须为正数")
        if (
            isinstance(self.localization_timeout_ms, bool)
            or self.localization_timeout_ms <= 0
        ):
            raise ValueError("重定位超时必须为正数")
        if (
            isinstance(self.max_recovery_attempts, bool)
            or self.max_recovery_attempts < 0
        ):
            raise ValueError("最大恢复次数不能为负数")
        if isinstance(self.arrival_confirmations, bool) or self.arrival_confirmations <= 0:
            raise ValueError("到达确认次数必须为正数")
        if self.max_recovery_attempts > 0 and not self.recovery_keys:
            raise ValueError("启用恢复时必须配置恢复按键")
        normalized_actions: dict[tuple[str, str], RouteAction] = {}
        for edge, raw_action in self.edge_actions.items():
            if (
                not isinstance(edge, tuple)
                or len(edge) != 2
                or not all(isinstance(node_id, str) and node_id for node_id in edge)
            ):
                raise ValueError("路线动作边必须由两个非空节点 ID 组成")
            action = RouteAction(key=raw_action) if isinstance(raw_action, str) else raw_action
            if not isinstance(action, RouteAction):
                raise ValueError("路线动作必须是 RouteAction 或兼容的按键字符串")
            normalized_actions[edge] = action
        object.__setattr__(self, "edge_actions", MappingProxyType(normalized_actions))
        object.__setattr__(self, "recovery_keys", tuple(self.recovery_keys))


@dataclass(frozen=True, slots=True)
class NavigationSnapshot:
    status: NavigationStatus
    route: tuple[str, ...]
    current_node_id: str | None
    next_node_id: str | None
    active_key: str | None
    recovery_attempts: int
    reason: str | None


class VisualNavigationController:
    """截图闭环控制器；卡住信号只能由锚点到 waypoint 的距离产生。"""

    def __init__(
        self,
        *,
        graph: Mapping[str, RouteNode],
        observer: WaypointObservationSource,
        actuator: _Actuator,
        goal_node_id: str,
        policy: NavigationPolicy,
    ) -> None:
        self._graph = dict(graph)
        self._observer = observer
        self._actuator = actuator
        self._goal_node_id = goal_node_id
        self._policy = policy
        self._status = NavigationStatus.LOCALIZING
        self._route: tuple[str, ...] = ()
        self._current_index = 0
        self._active_key: str | None = None
        self._pulse_deadline_ns: int | None = None
        self._prepared_edge: tuple[str, str] | None = None
        self._last_frame_sequence: int | None = None
        self._last_captured_at_ns: int | None = None
        self._localizing_since_ns: int | None = None
        self._last_progress_at_ns: int | None = None
        self._best_distance: float | None = None
        self._recovery_attempts = 0
        self._arrival_confirmations = 0
        self._reason: str | None = None

    def _current_node_id(self) -> str | None:
        if not self._route:
            return None
        return self._route[self._current_index]

    def _next_node_id(self) -> str | None:
        next_index = self._current_index + 1
        if next_index >= len(self._route):
            return None
        return self._route[next_index]

    def _observation_scope(self) -> ObservationScope:
        current_node_id = self._current_node_id()
        if current_node_id is None:
            return ObservationScope(allowed_waypoint_ids=None)
        next_node_id = self._next_node_id()
        return ObservationScope(
            allowed_waypoint_ids=frozenset(
                node_id
                for node_id in (current_node_id, next_node_id)
                if node_id is not None
            )
        )

    def _snapshot(self) -> NavigationSnapshot:
        return NavigationSnapshot(
            status=self._status,
            route=self._route,
            current_node_id=self._current_node_id(),
            next_node_id=self._next_node_id(),
            active_key=self._active_key,
            recovery_attempts=self._recovery_attempts,
            reason=self._reason,
        )

    def _release_active(self, *, now_ns: int, reason: str) -> None:
        if self._active_key is None:
            return
        self._actuator.key_up(self._active_key, now_ns=now_ns, reason=reason)
        self._active_key = None
        self._pulse_deadline_ns = None

    def _release_due(self, *, now_ns: int) -> None:
        if self._pulse_deadline_ns is None or now_ns < self._pulse_deadline_ns:
            return
        self._release_active(now_ns=now_ns, reason="动作脉冲到期")

    def _start_pulse(self, key: str, *, now_ns: int) -> None:
        if self._active_key is not None:
            return
        self._actuator.key_down(key, now_ns=now_ns)
        self._active_key = key
        self._pulse_deadline_ns = now_ns + self._policy.pulse_ms * 1_000_000

    def _start_route_action(
        self,
        edge: tuple[str, str],
        action: RouteAction,
        *,
        now_ns: int,
    ) -> None:
        if self._active_key is not None:
            return
        if self._prepared_edge != edge:
            if action.mouse_dx != 0 or action.mouse_dy != 0:
                self._actuator.move_mouse_relative(
                    action.mouse_dx,
                    action.mouse_dy,
                    now_ns=now_ns,
                )
            self._prepared_edge = edge
        self._start_pulse(action.key, now_ns=now_ns)

    def _stop_internal(self, *, now_ns: int, reason: str) -> NavigationSnapshot:
        if self._status in {NavigationStatus.ARRIVED, NavigationStatus.STOPPED}:
            return self._snapshot()
        self._actuator.release_all(now_ns=now_ns, reason=reason)
        self._active_key = None
        self._pulse_deadline_ns = None
        self._prepared_edge = None
        self._status = NavigationStatus.STOPPED
        self._reason = reason
        return self._snapshot()

    def _plan_from(self, waypoint_id: str, *, now_ns: int) -> bool:
        try:
            route = find_shortest_path(self._graph, waypoint_id, self._goal_node_id)
            for source_id, target_id in pairwise(route):
                if (source_id, target_id) not in self._policy.edge_actions:
                    raise ValueError(
                        f'路线边 "{source_id}->{target_id}" 缺少动作配置'
                    )
        except ValueError as error:
            self._stop_internal(now_ns=now_ns, reason=str(error))
            return False
        self._route = route
        self._current_index = 0
        self._status = NavigationStatus.NAVIGATING
        self._reason = None
        self._localizing_since_ns = None
        self._best_distance = None
        self._last_progress_at_ns = now_ns
        self._recovery_attempts = 0
        self._arrival_confirmations = 0
        return True

    def _enter_localizing(self, *, now_ns: int, reason: str) -> NavigationSnapshot:
        self._actuator.release_all(now_ns=now_ns, reason=reason)
        self._active_key = None
        self._pulse_deadline_ns = None
        self._status = NavigationStatus.LOCALIZING
        self._reason = reason
        if self._localizing_since_ns is None:
            self._localizing_since_ns = now_ns
        elapsed = now_ns - self._localizing_since_ns
        if elapsed >= self._policy.localization_timeout_ms * 1_000_000:
            return self._stop_internal(now_ns=now_ns, reason="视觉重定位超时")
        return self._snapshot()

    def _distance_to_next(self, centroid: tuple[float, float]) -> float:
        next_node_id = self._next_node_id()
        if next_node_id is None:
            return 0
        target = self._graph[next_node_id]
        return math.hypot(centroid[0] - target.x, centroid[1] - target.y)

    def _start_recovery(self, *, now_ns: int) -> NavigationSnapshot:
        self._actuator.release_all(now_ns=now_ns, reason="视觉进展超时")
        self._active_key = None
        self._pulse_deadline_ns = None
        if self._recovery_attempts >= self._policy.max_recovery_attempts:
            return self._stop_internal(now_ns=now_ns, reason="恢复次数已耗尽")
        key = self._policy.recovery_keys[
            self._recovery_attempts % len(self._policy.recovery_keys)
        ]
        self._recovery_attempts += 1
        self._status = NavigationStatus.RECOVERING
        self._reason = "视觉进展超时，执行有限恢复"
        self._start_pulse(key, now_ns=now_ns)
        return self._snapshot()

    def _handle_goal_confirmation(self, *, now_ns: int) -> NavigationSnapshot:
        self._actuator.release_all(now_ns=now_ns, reason="视觉确认目标节点")
        self._active_key = None
        self._pulse_deadline_ns = None
        self._prepared_edge = None
        self._arrival_confirmations += 1
        if self._arrival_confirmations < self._policy.arrival_confirmations:
            return self._snapshot()
        self._status = NavigationStatus.ARRIVED
        self._reason = "连续截图确认到达目标"
        return self._snapshot()

    def on_frame(self, frame: CapturedFrame, *, now_ns: int) -> NavigationSnapshot:
        if self._status in {NavigationStatus.ARRIVED, NavigationStatus.STOPPED}:
            return self._snapshot()
        self._release_due(now_ns=now_ns)
        if (
            self._last_frame_sequence is not None
            and (
                frame.sequence <= self._last_frame_sequence
                or frame.captured_at_ns <= (self._last_captured_at_ns or -1)
            )
        ):
            return self._stop_internal(now_ns=now_ns, reason="收到重复或过期截图帧")
        self._last_frame_sequence = frame.sequence
        self._last_captured_at_ns = frame.captured_at_ns

        observation = self._observer.observe(
            frame,
            scope=self._observation_scope(),
        )
        if observation.scope_violation or observation.out_of_scope_waypoint_id is not None:
            detail = (
                f'的非相邻 waypoint: "{observation.out_of_scope_waypoint_id}"'
                if observation.out_of_scope_waypoint_id is not None
                else ""
            )
            return self._stop_internal(
                now_ns=now_ns,
                reason=f"视觉定位跳到了观察范围外{detail}",
            )
        if observation.centroid is None:
            self._arrival_confirmations = 0
            return self._enter_localizing(now_ns=now_ns, reason="视觉锚点低置信或缺失")

        if self._status is NavigationStatus.LOCALIZING:
            if observation.waypoint_id is None:
                return self._enter_localizing(now_ns=now_ns, reason="无法唯一定位 waypoint")
            if not self._plan_from(observation.waypoint_id, now_ns=now_ns):
                return self._snapshot()

        current_node_id = self._current_node_id()
        next_node_id = self._next_node_id()
        if observation.waypoint_id == self._goal_node_id and current_node_id == self._goal_node_id:
            return self._handle_goal_confirmation(now_ns=now_ns)
        if next_node_id is not None and observation.waypoint_id == next_node_id:
            self._release_active(now_ns=now_ns, reason="视觉确认下一 waypoint")
            self._current_index += 1
            self._best_distance = None
            self._last_progress_at_ns = now_ns
            self._recovery_attempts = 0
            current_node_id = self._current_node_id()
            next_node_id = self._next_node_id()
            if current_node_id == self._goal_node_id:
                return self._handle_goal_confirmation(now_ns=now_ns)
        elif observation.waypoint_id not in {None, current_node_id}:
            return self._stop_internal(now_ns=now_ns, reason="视觉定位跳到了非相邻 waypoint")
        else:
            self._arrival_confirmations = 0

        if next_node_id is None:
            return self._snapshot()
        distance = self._distance_to_next(observation.centroid)
        progressed = (
            self._best_distance is None
            or self._best_distance - distance >= self._policy.min_progress_px
        )
        if progressed:
            if self._status is NavigationStatus.RECOVERING:
                self._release_active(now_ns=now_ns, reason="视觉进展已恢复")
            self._status = NavigationStatus.NAVIGATING
            self._reason = None
            self._best_distance = distance
            self._last_progress_at_ns = now_ns
            self._recovery_attempts = 0
        elif (
            self._last_progress_at_ns is not None
            and now_ns - self._last_progress_at_ns
            >= self._policy.stuck_after_ms * 1_000_000
        ):
            return self._start_recovery(now_ns=now_ns)

        edge = (current_node_id, next_node_id)
        action = self._policy.edge_actions[edge]
        self._start_route_action(edge, action, now_ns=now_ns)
        return self._snapshot()

    def on_timer(self, *, now_ns: int) -> NavigationSnapshot:
        if self._status in {NavigationStatus.ARRIVED, NavigationStatus.STOPPED}:
            return self._snapshot()
        self._release_due(now_ns=now_ns)
        if (
            self._status is NavigationStatus.LOCALIZING
            and self._localizing_since_ns is not None
            and now_ns - self._localizing_since_ns
            >= self._policy.localization_timeout_ms * 1_000_000
        ):
            return self._stop_internal(now_ns=now_ns, reason="视觉重定位超时")
        return self._snapshot()

    def stop(self, *, now_ns: int, reason: str) -> NavigationSnapshot:
        return self._stop_internal(now_ns=now_ns, reason=reason)
