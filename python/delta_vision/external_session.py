"""外部循环与仓库清理的生产配置、CLI 和安全装配入口。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .external_loop import (
    ExternalLoopResult,
    ExternalLoopSettings,
    ExternalLoopStatus,
    run_external_loop,
)
from .game_session import GameSessionSettings, load_game_session_settings
from .menu_profile import LoadedMenuProfile, load_menu_profile
from .passive_scene import PassiveReturnPolicy
from .sample_frames import validate_run_id
from .warehouse_cleanup import WarehouseCleanupPolicy
from .warehouse_worker import (
    WindowsWarehouseCleanupSession,
    WindowsWarehouseCleanupSettings,
)


@dataclass(frozen=True, slots=True)
class ExternalSessionSettings:
    """一次完整外部循环及其仓库清理配置。"""

    external_loop: ExternalLoopSettings
    warehouse_cleanup: WindowsWarehouseCleanupSettings

    def __post_init__(self) -> None:
        if not isinstance(self.external_loop, ExternalLoopSettings):
            raise TypeError("external_loop 必须是 ExternalLoopSettings")
        if not isinstance(
            self.warehouse_cleanup,
            WindowsWarehouseCleanupSettings,
        ):
            raise TypeError("warehouse_cleanup 必须是 WindowsWarehouseCleanupSettings")
        entry = self.external_loop.entry
        match = self.external_loop.match
        cleanup = self.warehouse_cleanup
        if entry.capture_backend != match.capture_backend:
            raise ValueError("外部循环 entry 和 match 的截图后端必须一致")
        if entry.max_key_hold_ms != match.max_key_hold_ms:
            raise ValueError("外部循环 entry 和 match 的最大按键保持时长必须一致")
        if entry.target_window_title != cleanup.target_window_title:
            raise ValueError("外部循环和仓库清理的目标窗口标题必须完全一致")
        if entry.capture_backend != cleanup.capture_backend:
            raise ValueError("外部循环和仓库清理的截图后端必须一致")
        if entry.emergency_virtual_key != cleanup.emergency_virtual_key:
            raise ValueError("外部循环和仓库清理的急停键必须一致")
        if entry.max_key_hold_ms != cleanup.max_key_hold_ms:
            raise ValueError("外部循环和仓库清理的最大按键保持时长必须一致")
        if entry.menu_profile.frame_size != cleanup.menu_profile.frame_size:
            raise ValueError("外部循环和仓库清理 Profile 的分辨率必须一致")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"外部会话配置包含重复字段: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"外部会话配置包含无效常量: {value}")


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


def _boolean(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'配置字段 "{field}" 必须是布尔值')
    return value


def _load_config(path: str | Path) -> tuple[Path, Mapping[str, object]]:
    config_path = Path(path).resolve()
    payload = config_path.read_bytes()
    try:
        text = payload.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError("外部会话配置必须是严格 UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"外部会话配置不是有效 JSON: {error}") from error
    raw = _mapping(parsed, field="root")
    if raw.get("schema_version") != 2:
        raise ValueError("外部会话只支持 schema_version=2")
    return config_path, raw


def _project_root(config_path: Path) -> Path:
    parent = config_path.parent.resolve()
    return parent.parent if parent.name.casefold() == "configs" else parent


def _profile_path(
    config_path: Path,
    value: object,
) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("仓库 Profile 必须是项目内相对路径")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or bool(windows.drive):
        raise ValueError("仓库 Profile 必须是项目内相对路径")
    candidate = (config_path.parent / value.replace("\\", "/")).resolve()
    try:
        candidate.relative_to(_project_root(config_path))
    except ValueError as error:
        raise ValueError("仓库 Profile 必须是项目内相对路径") from error
    return candidate


def _warehouse_policy(profile: LoadedMenuProfile) -> WarehouseCleanupPolicy:
    return WarehouseCleanupPolicy(
        confirmation_frames=profile.confirmation_frames,
        maximum_confirmation_span_ms=profile.maximum_confirmation_span_ms,
        maximum_frame_age_ms=profile.maximum_frame_age_ms,
        transition_timeout_ms=profile.transition_timeout_ms,
        maximum_action_point_drift_px=profile.maximum_action_point_drift_px,
        expected_frame_size=profile.frame_size,
    )


def _warehouse_settings_from_raw(
    config_path: Path,
    raw: Mapping[str, object],
    *,
    menu_profile_loader: Callable[[str | Path], LoadedMenuProfile],
) -> WindowsWarehouseCleanupSettings:
    title = raw.get("target_window_title")
    if not isinstance(title, str) or not title:
        raise ValueError("target_window_title 必须是非空字符串")
    backend = raw.get("capture_backend")
    if backend not in {"dxcam", "mss"}:
        raise ValueError('capture_backend 只能是 "dxcam" 或 "mss"')
    cleanup = _mapping(raw.get("warehouse_cleanup"), field="warehouse_cleanup")
    profile = menu_profile_loader(_profile_path(config_path, cleanup.get("profile")))
    return WindowsWarehouseCleanupSettings(
        target_window_title=title,
        capture_backend=backend,
        emergency_virtual_key=_positive_int(
            raw.get("emergency_virtual_key"),
            field="emergency_virtual_key",
        ),
        max_key_hold_ms=_positive_int(
            raw.get("max_key_hold_ms"),
            field="max_key_hold_ms",
        ),
        menu_profile=profile,
        loop_interval_ms=_positive_int(
            cleanup.get("loop_interval_ms"),
            field="warehouse_cleanup.loop_interval_ms",
        ),
        max_duration_seconds=_positive_number(
            cleanup.get("max_duration_seconds"),
            field="warehouse_cleanup.max_duration_seconds",
        ),
        armed_ready=_boolean(
            cleanup.get("armed_ready"),
            field="warehouse_cleanup.armed_ready",
        ),
        policy=_warehouse_policy(profile),
    )


def load_warehouse_cleanup_settings(
    path: str | Path,
    *,
    menu_profile_loader: Callable[[str | Path], LoadedMenuProfile] = load_menu_profile,
) -> WindowsWarehouseCleanupSettings:
    """只加载仓库链路；不要求尚未生成的路线 Profile 存在。"""

    config_path, raw = _load_config(path)
    return _warehouse_settings_from_raw(
        config_path,
        raw,
        menu_profile_loader=menu_profile_loader,
    )


def _return_policy(raw: Mapping[str, object]) -> PassiveReturnPolicy:
    external = _mapping(raw.get("external_loop"), field="external_loop")
    return_config = _mapping(external.get("return"), field="external_loop.return")
    return PassiveReturnPolicy(
        confirmation_frames=_positive_int(
            return_config.get("confirmation_frames"),
            field="external_loop.return.confirmation_frames",
        ),
        maximum_confirmation_span_ms=_positive_int(
            return_config.get("maximum_confirmation_span_ms"),
            field="external_loop.return.maximum_confirmation_span_ms",
        ),
        maximum_frame_age_ms=_positive_int(
            return_config.get("maximum_frame_age_ms"),
            field="external_loop.return.maximum_frame_age_ms",
        ),
        loop_interval_ms=_positive_int(
            return_config.get("loop_interval_ms"),
            field="external_loop.return.loop_interval_ms",
        ),
        max_duration_seconds=_positive_number(
            return_config.get("max_duration_seconds"),
            field="external_loop.return.max_duration_seconds",
        ),
    )


def load_external_session_settings(
    path: str | Path,
    *,
    game_session_loader: Callable[[str | Path], GameSessionSettings] = (
        load_game_session_settings
    ),
    menu_profile_loader: Callable[[str | Path], LoadedMenuProfile] = load_menu_profile,
) -> ExternalSessionSettings:
    config_path, raw = _load_config(path)
    game = game_session_loader(config_path)
    external = _mapping(raw.get("external_loop"), field="external_loop")
    cleanup = _warehouse_settings_from_raw(
        config_path,
        raw,
        menu_profile_loader=menu_profile_loader,
    )
    loop = ExternalLoopSettings(
        entry=game.menu_runtime_settings(),
        match=game.worker,
        return_policy=_return_policy(raw),
        cycle_limit=_positive_int(
            external.get("cycle_limit"),
            field="external_loop.cycle_limit",
        ),
    )
    return ExternalSessionSettings(
        external_loop=loop,
        warehouse_cleanup=cleanup,
    )


def run_windows_external_session(
    settings: ExternalSessionSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str,
    external_runner: Callable[..., ExternalLoopResult] = run_external_loop,
) -> ExternalLoopResult:
    """在创建 artifact 或解析窗口前一次性校验两项 Armed 授权。"""

    if not isinstance(settings, ExternalSessionSettings):
        raise TypeError("settings 必须是 ExternalSessionSettings")
    if type(armed) is not bool:
        raise ValueError("armed 必须是布尔值")
    if not callable(external_runner):
        raise TypeError("external_runner 必须可调用")
    parsed_run_id = validate_run_id(run_id)
    if armed and not settings.external_loop.match.armed_ready:
        raise ValueError("路线 armed_ready=false，禁止启动完整外循环")
    if armed and not settings.warehouse_cleanup.armed_ready:
        raise ValueError("仓库 armed_ready=false，禁止启动完整外循环")
    return external_runner(
        settings.external_loop,
        artifacts=artifacts,
        armed=armed,
        run_id=parsed_run_id,
        cleanup_session=WindowsWarehouseCleanupSession(
            settings=settings.warehouse_cleanup
        ),
    )


def _full_validation_payload(
    settings: ExternalSessionSettings,
    *,
    config: Path,
) -> dict[str, object]:
    loop = settings.external_loop
    return {
        "status": "valid",
        "mode": "external_loop",
        "config": str(config),
        "target_window_title": loop.entry.target_window_title,
        "entry_profile_id": loop.entry.menu_profile.profile_id,
        "entry_frame_size": list(loop.entry.menu_profile.frame_size),
        "route_frame_size": list(loop.match.perception.frame_size),
        "warehouse_profile_id": settings.warehouse_cleanup.menu_profile.profile_id,
        "warehouse_frame_size": list(
            settings.warehouse_cleanup.menu_profile.frame_size
        ),
        "route_armed_ready": loop.match.armed_ready,
        "warehouse_armed_ready": settings.warehouse_cleanup.armed_ready,
        "cycle_limit": loop.cycle_limit,
    }


def _cleanup_validation_payload(
    settings: WindowsWarehouseCleanupSettings,
    *,
    config: Path,
) -> dict[str, object]:
    return {
        "status": "valid",
        "mode": "warehouse_cleanup",
        "config": str(config),
        "target_window_title": settings.target_window_title,
        "warehouse_profile_id": settings.menu_profile.profile_id,
        "warehouse_frame_size": list(settings.menu_profile.frame_size),
        "warehouse_armed_ready": settings.armed_ready,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows 外部循环与仓库清理入口")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--confirm-armed", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.armed and not args.validate_only and not args.confirm_armed:
        print("Armed 必须显式传入 --confirm-armed；F12 是急停键。", file=sys.stderr)
        return 1
    artifacts = args.artifacts or Path("artifacts/runs") / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    run_id = args.run_id or uuid.uuid4().hex
    try:
        if args.cleanup_only:
            cleanup_settings = load_warehouse_cleanup_settings(args.config)
            if args.validate_only:
                print(
                    json.dumps(
                        _cleanup_validation_payload(
                            cleanup_settings,
                            config=args.config,
                        ),
                        ensure_ascii=False,
                    )
                )
                return 0
            cleanup = WindowsWarehouseCleanupSession(settings=cleanup_settings).run(
                artifacts=artifacts,
                armed=args.armed,
                run_id=run_id,
            )
            payload = dict(cleanup.summary)
            payload["artifacts"] = str(artifacts)
            print(json.dumps(payload, ensure_ascii=False))
            return 0 if cleanup.completed else 2

        settings = load_external_session_settings(args.config)
        if args.validate_only:
            print(
                json.dumps(
                    _full_validation_payload(settings, config=args.config),
                    ensure_ascii=False,
                )
            )
            return 0
        result = run_windows_external_session(
            settings,
            artifacts=artifacts,
            armed=args.armed,
            run_id=run_id,
        )
    except Exception as error:
        print(f"外部会话启动或运行失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": str(result.status),
                "completed_cycles": result.completed_cycles,
                "artifacts": str(artifacts),
                "armed": args.armed,
                "run_id": run_id,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.status is ExternalLoopStatus.COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
