"""外部视觉 Worker 的配置解析与可测试控制循环。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .actuator import DryRunActuator
from .capture import DxcamFrameSource, MssFrameSource
from .config import CaptureRegion
from .events import JsonlEventWriter, RuntimeEvent
from .frames import CapturedFrame, FrameRecorder
from .navigation import (
    NavigationPolicy,
    NavigationSnapshot,
    NavigationStatus,
    VisualNavigationController,
    WaypointObserver,
)
from .perception import ColorAnchorDetector
from .planner import RouteEdge, RouteNode
from .safe_input import SafetyGate, Win32InputActuator
from .win32_native import (
    Win32NativeGateway,
    find_window_handle,
    window_client_region,
)

SCAN_CODES = {
    "w": 0x11,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
    "e": 0x12,
    "shift": 0x2A,
    "space": 0x39,
}


class _FrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


class _Actuator(Protocol):
    @property
    def pressed_keys(self) -> frozenset[str]: ...

    def release_all(self, *, now_ns: int, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    target_window_title: str
    capture_backend: str
    emergency_virtual_key: int
    max_key_hold_ms: int
    loop_interval_ms: int
    max_duration_seconds: float
    marker_bgr: tuple[int, int, int]
    marker_tolerance: int
    marker_minimum_area: int
    marker_confidence_threshold: float
    localization_radius: float
    graph: Mapping[str, RouteNode]
    goal_node_id: str
    policy: NavigationPolicy


@dataclass(frozen=True, slots=True)
class ControlLoopResult:
    status: NavigationStatus
    frame_count: int
    duration_ns: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class WindowsRuntime:
    source: _FrameSource
    controller: VisualNavigationController
    actuator: DryRunActuator | Win32InputActuator
    recorder: FrameRecorder
    event_writer: JsonlEventWriter
    target_window_handle: int


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'配置字段 "{field}" 必须是对象')
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'配置字段 "{field}" 必须是正整数')
    return value


def _positive_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f'配置字段 "{field}" 必须是正有限数')
    return float(value)


def _finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f'配置字段 "{field}" 必须是有限数')
    return float(value)


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'配置字段 "{field}" 必须是非负整数')
    return value


def _parse_graph(raw_nodes: object) -> dict[str, RouteNode]:
    nodes = _mapping(raw_nodes, field="nodes")
    graph: dict[str, RouteNode] = {}
    for node_id, raw_node in nodes.items():
        if not node_id:
            raise ValueError("路线节点 ID 不能为空")
        node = _mapping(raw_node, field=f"nodes.{node_id}")
        raw_edges = node.get("edges")
        if not isinstance(raw_edges, list):
            raise ValueError(f'配置字段 "nodes.{node_id}.edges" 必须是数组')
        edges = []
        for index, raw_edge in enumerate(raw_edges):
            edge = _mapping(raw_edge, field=f"nodes.{node_id}.edges[{index}]")
            target = edge.get("target")
            if not isinstance(target, str) or not target:
                raise ValueError("路线边 target 必须是非空字符串")
            edges.append(
                RouteEdge(
                    target_node_id=target,
                    cost=_finite_number(edge.get("cost"), field="edge.cost"),
                )
            )
            if edges[-1].cost < 0:
                raise ValueError("路线边 cost 不能为负数")
        graph[node_id] = RouteNode(
            x=_finite_number(node.get("x"), field=f"nodes.{node_id}.x"),
            y=_finite_number(node.get("y"), field=f"nodes.{node_id}.y"),
            edges=tuple(edges),
        )
    return graph


def _parse_edge_actions(raw_actions: object) -> dict[tuple[str, str], str]:
    if not isinstance(raw_actions, list):
        raise ValueError('配置字段 "edge_actions" 必须是数组')
    actions: dict[tuple[str, str], str] = {}
    for index, raw_action in enumerate(raw_actions):
        action = _mapping(raw_action, field=f"edge_actions[{index}]")
        source = action.get("source")
        target = action.get("target")
        key = action.get("key")
        if not all(isinstance(value, str) and value for value in (source, target, key)):
            raise ValueError("路线动作的 source、target、key 必须是非空字符串")
        edge = (source, target)
        if edge in actions:
            raise ValueError(f'路线动作重复: "{source}->{target}"')
        actions[edge] = key
    return actions


def load_worker_settings(path: str | Path) -> WorkerSettings:
    config_path = Path(path)
    raw = _mapping(json.loads(config_path.read_text(encoding="utf-8")), field="root")
    if raw.get("schema_version") != 1:
        raise ValueError("只支持 schema_version=1 的 Worker 配置")
    title = raw.get("target_window_title")
    if not isinstance(title, str) or not title:
        raise ValueError("target_window_title 必须是非空字符串")
    backend = raw.get("capture_backend")
    if backend not in {"dxcam", "mss"}:
        raise ValueError('capture_backend 只能是 "dxcam" 或 "mss"')
    marker = _mapping(raw.get("marker"), field="marker")
    raw_bgr = marker.get("bgr")
    if (
        not isinstance(raw_bgr, list)
        or len(raw_bgr) != 3
        or any(type(channel) is not int or not 0 <= channel <= 255 for channel in raw_bgr)
    ):
        raise ValueError("marker.bgr 必须包含三个 0 到 255 的整数")
    graph = _parse_graph(raw.get("nodes"))
    goal_node_id = raw.get("goal_node_id")
    if not isinstance(goal_node_id, str) or not goal_node_id:
        raise ValueError("goal_node_id 必须是非空字符串")
    if goal_node_id not in graph:
        raise ValueError(f'goal_node_id 不在路线图中: "{goal_node_id}"')
    navigation = _mapping(raw.get("navigation"), field="navigation")
    raw_recovery_keys = navigation.get("recovery_keys")
    if not isinstance(raw_recovery_keys, list) or not all(
        isinstance(key, str) and key for key in raw_recovery_keys
    ):
        raise ValueError("navigation.recovery_keys 必须是字符串数组")
    policy = NavigationPolicy(
        edge_actions=_parse_edge_actions(raw.get("edge_actions")),
        pulse_ms=_positive_int(navigation.get("pulse_ms"), field="navigation.pulse_ms"),
        min_progress_px=_positive_number(
            navigation.get("min_progress_px"), field="navigation.min_progress_px"
        ),
        stuck_after_ms=_positive_int(
            navigation.get("stuck_after_ms"), field="navigation.stuck_after_ms"
        ),
        localization_timeout_ms=_positive_int(
            navigation.get("localization_timeout_ms"),
            field="navigation.localization_timeout_ms",
        ),
        max_recovery_attempts=_non_negative_int(
            navigation.get("max_recovery_attempts"),
            field="navigation.max_recovery_attempts",
        ),
        recovery_keys=tuple(raw_recovery_keys),
        arrival_confirmations=_positive_int(
            navigation.get("arrival_confirmations"),
            field="navigation.arrival_confirmations",
        ),
    )
    confidence_threshold = _positive_number(
        marker.get("confidence_threshold"),
        field="marker.confidence_threshold",
    )
    if confidence_threshold > 1:
        raise ValueError("marker.confidence_threshold 不能超过 1")
    emergency_virtual_key = _positive_int(
        raw.get("emergency_virtual_key"), field="emergency_virtual_key"
    )
    if emergency_virtual_key > 255:
        raise ValueError("emergency_virtual_key 不能超过 255")
    max_key_hold_ms = _positive_int(
        raw.get("max_key_hold_ms"), field="max_key_hold_ms"
    )
    if policy.pulse_ms > max_key_hold_ms:
        raise ValueError("navigation.pulse_ms 不能超过 max_key_hold_ms")
    return WorkerSettings(
        target_window_title=title,
        capture_backend=backend,
        emergency_virtual_key=emergency_virtual_key,
        max_key_hold_ms=max_key_hold_ms,
        loop_interval_ms=_positive_int(
            raw.get("loop_interval_ms"), field="loop_interval_ms"
        ),
        max_duration_seconds=_positive_number(
            raw.get("max_duration_seconds"), field="max_duration_seconds"
        ),
        marker_bgr=tuple(raw_bgr),
        marker_tolerance=_non_negative_int(
            marker.get("tolerance"), field="marker.tolerance"
        ),
        marker_minimum_area=_positive_int(
            marker.get("minimum_area"), field="marker.minimum_area"
        ),
        marker_confidence_threshold=confidence_threshold,
        localization_radius=_positive_number(
            raw.get("localization_radius"), field="localization_radius"
        ),
        graph=graph,
        goal_node_id=goal_node_id,
        policy=policy,
    )


def build_navigation_controller(
    settings: WorkerSettings,
    *,
    actuator: DryRunActuator | Win32InputActuator,
) -> VisualNavigationController:
    detector = ColorAnchorDetector(
        label="player",
        bgr=settings.marker_bgr,
        tolerance=settings.marker_tolerance,
        minimum_area=settings.marker_minimum_area,
        confidence_threshold=settings.marker_confidence_threshold,
    )
    observer = WaypointObserver(
        detector=detector,
        waypoint_positions={
            node_id: (node.x, node.y) for node_id, node in settings.graph.items()
        },
        localization_radius=settings.localization_radius,
    )
    return VisualNavigationController(
        graph=settings.graph,
        observer=observer,
        actuator=actuator,
        goal_node_id=settings.goal_node_id,
        policy=settings.policy,
    )


def build_windows_runtime(
    settings: WorkerSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    window_handle_resolver: Callable[[str], int] = find_window_handle,
    region_resolver: Callable[[str], CaptureRegion] = window_client_region,
    dxcam_factory: Callable[[CaptureRegion], _FrameSource] = DxcamFrameSource,
    mss_factory: Callable[[CaptureRegion], _FrameSource] = MssFrameSource,
    gateway_factory: Callable[[], Win32NativeGateway] = Win32NativeGateway,
) -> WindowsRuntime:
    allowed_keys = set(settings.policy.edge_actions.values()) | set(
        settings.policy.recovery_keys
    )
    unsupported_keys = allowed_keys - SCAN_CODES.keys()
    if unsupported_keys:
        raise ValueError(f"配置包含不支持的按键: {sorted(unsupported_keys)}")
    # region_resolver 会先建立 DPI Awareness，随后解析的 HWND 与 DXGI 使用同一坐标系。
    region = region_resolver(settings.target_window_title)
    target_window_handle = window_handle_resolver(settings.target_window_title)
    source_factory = dxcam_factory if settings.capture_backend == "dxcam" else mss_factory
    source = source_factory(region)
    try:
        if armed:
            gateway = gateway_factory()
            gate = SafetyGate(
                target_window_title=settings.target_window_title,
                target_window_handle=target_window_handle,
                emergency_virtual_key=settings.emergency_virtual_key,
                gateway=gateway,
            )
            actuator: DryRunActuator | Win32InputActuator = Win32InputActuator(
                scan_codes={key: SCAN_CODES[key] for key in allowed_keys},
                max_key_hold_ms=settings.max_key_hold_ms,
                gate=gate,
                gateway=gateway,
            )
        else:
            actuator = DryRunActuator(
                allowed_keys=allowed_keys,
                max_key_hold_ms=settings.max_key_hold_ms,
            )
        controller = build_navigation_controller(settings, actuator=actuator)
        artifact_root = Path(artifacts)
        return WindowsRuntime(
            source=source,
            controller=controller,
            actuator=actuator,
            recorder=FrameRecorder(artifact_root / "replay"),
            event_writer=JsonlEventWriter(artifact_root / "events.jsonl"),
            target_window_handle=target_window_handle,
        )
    except BaseException:
        source.close()
        raise


def _snapshot_payload(
    snapshot: NavigationSnapshot, *, pressed_keys: frozenset[str]
) -> dict[str, object]:
    return {
        "status": str(snapshot.status),
        "route": list(snapshot.route),
        "current_node_id": snapshot.current_node_id,
        "next_node_id": snapshot.next_node_id,
        "active_key": snapshot.active_key,
        "recovery_attempts": snapshot.recovery_attempts,
        "reason": snapshot.reason,
        "pressed_keys": sorted(pressed_keys),
    }


def run_control_loop(
    *,
    source: _FrameSource,
    controller: VisualNavigationController,
    actuator: _Actuator,
    recorder: FrameRecorder,
    event_writer: JsonlEventWriter,
    loop_interval_ms: int,
    max_duration_seconds: float,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ControlLoopResult:
    started_at_ns = clock_ns()
    frame_count = 0
    snapshot: NavigationSnapshot | None = None
    try:
        while True:
            now_ns = clock_ns()
            if now_ns - started_at_ns >= max_duration_seconds * 1_000_000_000:
                snapshot = controller.stop(now_ns=now_ns, reason="Worker 运行超时")
                break
            frame = source.grab()
            if frame is None:
                snapshot = controller.on_timer(now_ns=now_ns)
            else:
                snapshot = controller.on_frame(frame, now_ns=now_ns)
                payload = _snapshot_payload(
                    snapshot, pressed_keys=actuator.pressed_keys
                )
                recorder.record(frame, metadata={"navigation": payload})
                event_writer.write(
                    RuntimeEvent(event_type="frame", at_ns=now_ns, payload=payload)
                )
                frame_count += 1
            if snapshot.status in {NavigationStatus.ARRIVED, NavigationStatus.STOPPED}:
                break
            sleep_fn(loop_interval_ms / 1_000)
        ended_at_ns = clock_ns()
        return ControlLoopResult(
            status=snapshot.status,
            frame_count=frame_count,
            duration_ns=max(0, ended_at_ns - started_at_ns),
            reason=snapshot.reason,
        )
    except BaseException:
        controller.stop(now_ns=clock_ns(), reason="Worker 异常，执行安全停止")
        raise
    finally:
        source.close()


def run_windows_worker(
    settings: WorkerSettings,
    *,
    artifacts: str | Path,
    armed: bool,
) -> ControlLoopResult:
    runtime = build_windows_runtime(settings, artifacts=artifacts, armed=armed)
    return run_control_loop(
        source=runtime.source,
        controller=runtime.controller,
        actuator=runtime.actuator,
        recorder=runtime.recorder,
        event_writer=runtime.event_writer,
        loop_interval_ms=settings.loop_interval_ms,
        max_duration_seconds=settings.max_duration_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows 纯外部截图视觉导航 Worker")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled-window.json"),
    )
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument(
        "--armed",
        action="store_true",
        help="显式启用标准 SendInput；默认只记录动作，不发送输入",
    )
    args = parser.parse_args(argv)
    artifacts = args.artifacts or Path("artifacts/runs") / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    try:
        settings = load_worker_settings(args.config)
        result = run_windows_worker(settings, artifacts=artifacts, armed=args.armed)
    except Exception as error:
        print(f"Worker 启动或运行失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": str(result.status),
                "frame_count": result.frame_count,
                "duration_ns": result.duration_ns,
                "reason": result.reason,
                "artifacts": str(artifacts),
                "armed": args.armed,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.status is NavigationStatus.ARRIVED else 2


if __name__ == "__main__":
    raise SystemExit(main())
