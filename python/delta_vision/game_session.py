"""从大厅菜单确认到局内路线的单进程全会话编排。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from .actuator import DryRunActuator
from .capture import DxcamFrameSource, MssFrameSource
from .config import CaptureRegion
from .events import JsonlEventWriter
from .frames import FrameRecorder
from .menu_automation import MenuActionKind, MenuControllerStatus, MenuScene
from .menu_profile import LoadedMenuProfile, load_menu_profile
from .menu_runtime import MenuActionExecutor
from .menu_worker import MenuLoopResult, run_menu_control_loop
from .navigation import NavigationStatus
from .safe_input import SafetyGate, Win32InputActuator
from .template_profile import TemplateProfile
from .win32_native import Win32NativeGateway, find_window_handle, window_client_region
from .worker import (
    SCAN_CODES,
    ControlLoopResult,
    WorkerSettings,
    load_worker_settings,
    run_windows_worker,
)


class _FrameSource(Protocol):
    def grab(self): ...

    def close(self) -> None: ...


class GameSessionStatus(StrEnum):
    COMPLETED = "completed"
    MENU_STOPPED = "menu_stopped"
    ROUTE_STOPPED = "route_stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class GameSessionSettings:
    worker: WorkerSettings
    menu_profile: LoadedMenuProfile
    menu_loop_interval_ms: int
    menu_max_duration_seconds: float

    def __post_init__(self) -> None:
        if type(self.menu_loop_interval_ms) is not int or self.menu_loop_interval_ms <= 0:
            raise ValueError("menu.loop_interval_ms 必须是正整数")
        if (
            isinstance(self.menu_max_duration_seconds, bool)
            or not isinstance(self.menu_max_duration_seconds, (int, float))
            or not math.isfinite(self.menu_max_duration_seconds)
            or self.menu_max_duration_seconds <= 0
        ):
            raise ValueError("menu.max_duration_seconds 必须是正有限数")
        transitions = self.menu_profile.transitions
        if (
            not transitions
            or transitions[0].source is not MenuScene.LOBBY
            or transitions[-1].target is not MenuScene.IN_MATCH
        ):
            raise ValueError("全会话菜单流程必须从 LOBBY 连续确认到 IN_MATCH")
        menu_keys = {
            transition.key
            for transition in transitions
            if transition.action_kind is MenuActionKind.KEY
        }
        unsupported_keys = menu_keys - SCAN_CODES.keys()
        if unsupported_keys:
            raise ValueError(f"配置包含不支持的菜单按键: {sorted(unsupported_keys)}")
        if not isinstance(self.worker.perception, TemplateProfile):
            raise ValueError(
                "全会话只接受 schema_version=2 的 TemplateProfile 路线配置"
            )
        if self.worker.perception.frame_size != self.menu_profile.frame_size:
            raise ValueError("菜单与局内路线 Profile 的期望分辨率必须一致")


@dataclass(frozen=True, slots=True)
class WindowsMenuRuntime:
    source: _FrameSource
    observer: object
    controller: object
    executor: MenuActionExecutor
    actuator: DryRunActuator | Win32InputActuator
    recorder: FrameRecorder
    event_writer: JsonlEventWriter
    target_window_handle: int


@dataclass(frozen=True, slots=True)
class GameSessionResult:
    status: GameSessionStatus
    menu: MenuLoopResult
    route: ControlLoopResult | None


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
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


def _relative_path_reference(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f'配置字段 "{field}" 必须是非空相对路径')
    windows_value = PureWindowsPath(value)
    if (
        PurePosixPath(value).is_absolute()
        or windows_value.is_absolute()
        or bool(windows_value.drive)
    ):
        raise ValueError(f'配置字段 "{field}" 必须是相对路径')
    return value


def load_game_session_settings(
    path: str | Path,
    *,
    worker_loader: Callable[[str | Path], WorkerSettings] = load_worker_settings,
    menu_profile_loader: Callable[[str | Path], LoadedMenuProfile] = load_menu_profile,
) -> GameSessionSettings:
    config_path = Path(path).resolve()
    raw = _mapping(json.loads(config_path.read_text(encoding="utf-8")), field="root")
    menu = _mapping(raw.get("menu"), field="menu")
    reference = _relative_path_reference(menu.get("profile"), field="menu.profile")
    profile_path = (config_path.parent / reference.replace("\\", "/")).resolve()
    return GameSessionSettings(
        worker=worker_loader(config_path),
        menu_profile=menu_profile_loader(profile_path),
        menu_loop_interval_ms=_positive_int(
            menu.get("loop_interval_ms"),
            field="menu.loop_interval_ms",
        ),
        menu_max_duration_seconds=_positive_number(
            menu.get("max_duration_seconds"),
            field="menu.max_duration_seconds",
        ),
    )


def _allowed_keys(settings: GameSessionSettings) -> set[str]:
    route_keys = {
        action.key for action in settings.worker.policy.edge_actions.values()
    } | set(settings.worker.policy.recovery_keys)
    menu_keys = {
        transition.key
        for transition in settings.menu_profile.transitions
        if transition.action_kind is MenuActionKind.KEY and transition.key is not None
    }
    return route_keys | menu_keys


def build_windows_menu_runtime(
    settings: GameSessionSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str | None = None,
    window_handle_resolver: Callable[[str], int] = find_window_handle,
    region_resolver: Callable[[str], CaptureRegion] = window_client_region,
    dxcam_factory: Callable[[CaptureRegion], _FrameSource] = DxcamFrameSource,
    mss_factory: Callable[[CaptureRegion], _FrameSource] = MssFrameSource,
    gateway_factory: Callable[[], Win32NativeGateway] = Win32NativeGateway,
) -> WindowsMenuRuntime:
    if armed and not settings.worker.armed_ready:
        raise ValueError("全会话 armed 被配置 armed_ready=false 阻止")
    allowed_keys = _allowed_keys(settings)
    unsupported_keys = allowed_keys - SCAN_CODES.keys()
    if unsupported_keys:
        raise ValueError(f"配置包含不支持的按键: {sorted(unsupported_keys)}")
    region = region_resolver(settings.worker.target_window_title)
    expected_width, expected_height = settings.menu_profile.frame_size
    if (region.width, region.height) != (expected_width, expected_height):
        raise ValueError(
            "菜单 Profile 分辨率与目标窗口客户区不一致: "
            f"expected={expected_width}x{expected_height}, "
            f"actual={region.width}x{region.height}"
        )
    target_window_handle = window_handle_resolver(settings.worker.target_window_title)
    source_factory = (
        dxcam_factory if settings.worker.capture_backend == "dxcam" else mss_factory
    )
    source = source_factory(region)
    try:
        if armed:
            gateway = gateway_factory()
            gate = SafetyGate(
                target_window_title=settings.worker.target_window_title,
                target_window_handle=target_window_handle,
                emergency_virtual_key=settings.worker.emergency_virtual_key,
                gateway=gateway,
            )
            actuator: DryRunActuator | Win32InputActuator = Win32InputActuator(
                scan_codes={key: SCAN_CODES[key] for key in allowed_keys},
                max_key_hold_ms=settings.worker.max_key_hold_ms,
                gate=gate,
                gateway=gateway,
            )
        else:
            actuator = DryRunActuator(
                allowed_keys=allowed_keys,
                max_key_hold_ms=settings.worker.max_key_hold_ms,
            )
        artifact_root = Path(artifacts)
        return WindowsMenuRuntime(
            source=source,
            observer=settings.menu_profile.observer,
            controller=settings.menu_profile.create_controller(),
            executor=MenuActionExecutor(
                actuator=actuator,
                capture_region=region,
            ),
            actuator=actuator,
            recorder=FrameRecorder(artifact_root / "replay"),
            event_writer=JsonlEventWriter(
                artifact_root / "events.jsonl",
                run_id=run_id,
                truncate=True,
            ),
            target_window_handle=target_window_handle,
        )
    except BaseException:
        source.close()
        raise


def run_windows_menu(
    settings: GameSessionSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str | None = None,
) -> MenuLoopResult:
    runtime = build_windows_menu_runtime(
        settings,
        artifacts=artifacts,
        armed=armed,
        run_id=run_id,
    )
    return run_menu_control_loop(
        source=runtime.source,
        observer=runtime.observer,
        controller=runtime.controller,
        executor=runtime.executor,
        actuator=runtime.actuator,
        recorder=runtime.recorder,
        event_writer=runtime.event_writer,
        loop_interval_ms=settings.menu_loop_interval_ms,
        max_duration_seconds=settings.menu_max_duration_seconds,
    )


def _menu_payload(result: MenuLoopResult) -> dict[str, object]:
    return {
        "status": str(result.status),
        "frame_count": result.frame_count,
        "action_count": result.action_count,
        "duration_ns": result.duration_ns,
        "reason": result.reason,
    }


def _route_payload(result: ControlLoopResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "status": str(result.status),
        "frame_count": result.frame_count,
        "duration_ns": result.duration_ns,
        "reason": result.reason,
    }


def _write_summary(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_failure_summary(
    *,
    path: Path,
    run_id: str,
    armed: bool,
    failed_phase: str,
    error: BaseException,
    menu_result: MenuLoopResult | None,
) -> None:
    try:
        _write_summary(
            path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "armed": armed,
                "status": str(GameSessionStatus.FAILED),
                "failed_phase": failed_phase,
                "menu": None if menu_result is None else _menu_payload(menu_result),
                "route": None,
                "error": {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
    except BaseException as summary_error:
        error.add_note(
            "写入全会话失败摘要时出错: "
            f"{type(summary_error).__name__}: {summary_error}"
        )


def run_game_session(
    settings: GameSessionSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str,
    menu_runner: Callable[..., MenuLoopResult] = run_windows_menu,
    route_runner: Callable[..., ControlLoopResult] = run_windows_worker,
) -> GameSessionResult:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id 必须是非空字符串")
    if armed and not settings.worker.armed_ready:
        raise ValueError("全会话 armed 被配置 armed_ready=false 阻止")
    artifact_root = Path(artifacts)
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "session-summary.json"
    try:
        menu_result = menu_runner(
            settings=settings,
            artifacts=artifact_root / "menu",
            armed=armed,
            run_id=run_id,
        )
    except BaseException as error:
        _write_failure_summary(
            path=summary_path,
            run_id=run_id,
            armed=armed,
            failed_phase="menu",
            error=error,
            menu_result=None,
        )
        raise
    if menu_result.status is not MenuControllerStatus.COMPLETED:
        result = GameSessionResult(
            status=GameSessionStatus.MENU_STOPPED,
            menu=menu_result,
            route=None,
        )
        _write_summary(
            summary_path,
            {
                "schema_version": 1,
                "run_id": run_id,
                "armed": armed,
                "status": str(result.status),
                "menu": _menu_payload(menu_result),
                "route": None,
            },
        )
        return result

    try:
        route_result = route_runner(
            settings.worker,
            artifacts=artifact_root / "route",
            armed=armed,
            run_id=run_id,
        )
    except BaseException as error:
        _write_failure_summary(
            path=summary_path,
            run_id=run_id,
            armed=armed,
            failed_phase="route",
            error=error,
            menu_result=menu_result,
        )
        raise
    status = (
        GameSessionStatus.COMPLETED
        if route_result.status is NavigationStatus.ARRIVED
        else GameSessionStatus.ROUTE_STOPPED
    )
    result = GameSessionResult(status=status, menu=menu_result, route=route_result)
    _write_summary(
        summary_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "armed": armed,
            "status": str(status),
            "menu": _menu_payload(menu_result),
            "route": _route_payload(route_result),
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="大厅菜单到局内路线的全会话 Worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    artifacts = args.artifacts or Path("artifacts/runs") / time.strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or uuid.uuid4().hex
    try:
        settings = load_game_session_settings(args.config)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "config": str(args.config),
                        "menu_profile_id": settings.menu_profile.profile_id,
                        "target_window_title": settings.worker.target_window_title,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        result = run_game_session(
            settings,
            artifacts=artifacts,
            armed=args.armed,
            run_id=run_id,
        )
    except Exception as error:
        print(f"全会话启动或运行失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": str(result.status),
                "menu_status": str(result.menu.status),
                "route_status": None if result.route is None else str(result.route.status),
                "artifacts": str(artifacts),
                "armed": args.armed,
                "run_id": run_id,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.status is GameSessionStatus.COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
