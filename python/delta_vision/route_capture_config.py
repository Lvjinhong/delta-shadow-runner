"""路线标定采集配置和确定性计划。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .menu_profile import LoadedMenuProfile, load_menu_profile
from .sample_frames import validate_dataset_split
from .worker import SCAN_CODES

MAX_ROUTE_CAPTURE_STEPS = 256
MAX_MOUSE_DELTA = 4096
MAX_SETTLE_MS = 5_000
REQUIRED_FRAME_SIZE = (1920, 1080)
F12_VIRTUAL_KEY = 123
# 给 Windows Timer 调度和 SendInput 释放保留 50ms 余量，避免只靠控制线程轮询。
ROUTE_CAPTURE_WATCHDOG_MS = 150


@dataclass(frozen=True, slots=True)
class RouteCaptureStep:
    step_id: str
    keys: tuple[str, ...]
    pulse_ms: int
    mouse_dx: int
    mouse_dy: int
    settle_ms: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.step_id) is None
        ):
            raise ValueError("路线采集 step_id 格式非法")
        if (
            not isinstance(self.keys, tuple)
            or not 1 <= len(self.keys) <= 3
            or any(not isinstance(key, str) or not key for key in self.keys)
            or len(set(self.keys)) != len(self.keys)
        ):
            raise ValueError("路线采集 keys 必须包含 1 到 3 个唯一按键")
        unsupported = set(self.keys) - SCAN_CODES.keys()
        if unsupported:
            raise ValueError(f"路线采集包含不支持的按键: {sorted(unsupported)}")
        if type(self.pulse_ms) is not int or self.pulse_ms <= 0:
            raise ValueError("路线采集 pulse_ms 必须是正整数")
        if self.pulse_ms > ROUTE_CAPTURE_WATCHDOG_MS:
            raise ValueError(
                f"路线采集 pulse_ms 不能超过 {ROUTE_CAPTURE_WATCHDOG_MS}ms"
            )
        for value, field in ((self.mouse_dx, "mouse_dx"), (self.mouse_dy, "mouse_dy")):
            if type(value) is not int or abs(value) > MAX_MOUSE_DELTA:
                raise ValueError(f"路线采集 {field} 必须是允许范围内的整数")
        if (
            type(self.settle_ms) is not int
            or not 0 <= self.settle_ms <= MAX_SETTLE_MS
        ):
            raise ValueError("路线采集 settle_ms 超出允许范围")


@dataclass(frozen=True, slots=True)
class RouteCaptureSettings:
    target_window_title: str
    capture_backend: str
    emergency_virtual_key: int
    max_key_hold_ms: int
    menu_profile: LoadedMenuProfile
    armed_ready: bool
    dataset_split: str
    guard_interval_ms: int
    max_duration_seconds: float
    steps: tuple[RouteCaptureStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target_window_title, str) or not self.target_window_title:
            raise ValueError("target_window_title 必须是非空字符串")
        if self.capture_backend not in {"dxcam", "mss"}:
            raise ValueError('capture_backend 只能是 "dxcam" 或 "mss"')
        if self.emergency_virtual_key != F12_VIRTUAL_KEY:
            raise ValueError("路线采集 emergency_virtual_key 必须固定为 F12 (123)")
        if type(self.max_key_hold_ms) is not int or self.max_key_hold_ms <= 0:
            raise ValueError("max_key_hold_ms 必须是正整数")
        if type(self.armed_ready) is not bool:
            raise ValueError("route_capture.armed_ready 必须是布尔值")
        validate_dataset_split(self.dataset_split)
        if (
            type(self.guard_interval_ms) is not int
            or not 10 <= self.guard_interval_ms <= 50
        ):
            raise ValueError("route_capture.guard_interval_ms 必须是 10 到 50 的整数")
        if (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("route_capture.max_duration_seconds 必须是正有限数")
        if (
            not isinstance(self.steps, tuple)
            or not 1 <= len(self.steps) <= MAX_ROUTE_CAPTURE_STEPS
            or any(not isinstance(step, RouteCaptureStep) for step in self.steps)
        ):
            raise ValueError("route_capture.steps 必须是非空且有上限的动作序列")
        step_ids = tuple(step.step_id for step in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("route_capture.steps 的 step_id 不能重复")
        if any(step.pulse_ms > self.max_key_hold_ms for step in self.steps):
            raise ValueError("路线采集 pulse_ms 不能超过 max_key_hold_ms")
        if getattr(self.menu_profile, "frame_size", None) != REQUIRED_FRAME_SIZE:
            raise ValueError("路线采集只接受 1920x1080 菜单 Profile")
        for field in (
            "observer",
            "profile_id",
            "profile_sha256",
            "confirmation_frames",
            "maximum_confirmation_span_ms",
            "maximum_frame_age_ms",
        ):
            if not hasattr(self.menu_profile, field):
                raise ValueError(f"菜单 Profile 缺少路线采集字段: {field}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"路线采集配置包含重复字段: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"路线采集配置包含无效常量: {value}")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'配置字段 "{field}" 必须是对象')
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'配置字段 "{field}" 必须是正整数')
    return value


def _finite_positive(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f'配置字段 "{field}" 必须是正有限数')
    return float(value)


def _project_root(config_path: Path) -> Path:
    parent = config_path.parent.resolve()
    return parent.parent if parent.name.casefold() == "configs" else parent


def _profile_path(config_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("菜单 Profile 必须是项目内相对路径")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or bool(windows.drive):
        raise ValueError("菜单 Profile 必须是项目内相对路径")
    candidate = (config_path.parent / value.replace("\\", "/")).resolve()
    try:
        candidate.relative_to(_project_root(config_path))
    except ValueError as error:
        raise ValueError("菜单 Profile 必须是项目内相对路径") from error
    return candidate


def load_route_capture_settings(
    path: str | Path,
    *,
    menu_profile_loader: Callable[[str | Path], LoadedMenuProfile] = load_menu_profile,
) -> RouteCaptureSettings:
    config_path = Path(path).resolve()
    try:
        parsed = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError("路线采集配置必须是严格 UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"路线采集配置不是有效 JSON: {error}") from error
    raw = _mapping(parsed, field="root")
    if type(raw.get("schema_version")) is not int or raw.get("schema_version") != 2:
        raise ValueError("路线采集只支持 schema_version=2")
    title = raw.get("target_window_title")
    if not isinstance(title, str) or not title:
        raise ValueError("target_window_title 必须是非空字符串")
    backend = raw.get("capture_backend")
    if backend not in {"dxcam", "mss"}:
        raise ValueError('capture_backend 只能是 "dxcam" 或 "mss"')
    menu = _mapping(raw.get("menu"), field="menu")
    profile = menu_profile_loader(_profile_path(config_path, menu.get("profile")))
    capture = _mapping(raw.get("route_capture"), field="route_capture")
    raw_steps = capture.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("route_capture.steps 必须是非空数组")
    if len(raw_steps) > MAX_ROUTE_CAPTURE_STEPS:
        raise ValueError("route_capture.steps 超过允许上限")
    steps: list[RouteCaptureStep] = []
    for index, raw_step in enumerate(raw_steps):
        step = _mapping(raw_step, field=f"route_capture.steps[{index}]")
        keys = step.get("keys")
        if not isinstance(keys, list):
            raise ValueError(f"route_capture.steps[{index}].keys 必须是数组")
        steps.append(
            RouteCaptureStep(
                step_id=step.get("step_id"),
                keys=tuple(keys),
                pulse_ms=step.get("pulse_ms"),
                mouse_dx=step.get("mouse_dx", 0),
                mouse_dy=step.get("mouse_dy", 0),
                settle_ms=step.get("settle_ms", 0),
            )
        )
    armed_ready = capture.get("armed_ready")
    if type(armed_ready) is not bool:
        raise ValueError("route_capture.armed_ready 必须是布尔值")
    dataset_split = capture.get("dataset_split")
    if not isinstance(dataset_split, str):
        raise ValueError("route_capture.dataset_split 必须是字符串")
    return RouteCaptureSettings(
        target_window_title=title,
        capture_backend=backend,
        emergency_virtual_key=_positive_int(
            raw.get("emergency_virtual_key"), field="emergency_virtual_key"
        ),
        max_key_hold_ms=_positive_int(
            raw.get("max_key_hold_ms"), field="max_key_hold_ms"
        ),
        menu_profile=profile,
        armed_ready=armed_ready,
        dataset_split=dataset_split,
        guard_interval_ms=_positive_int(
            capture.get("guard_interval_ms"),
            field="route_capture.guard_interval_ms",
        ),
        max_duration_seconds=_finite_positive(
            capture.get("max_duration_seconds"),
            field="route_capture.max_duration_seconds",
        ),
        steps=tuple(steps),
    )


def canonical_route_capture_plan(settings: RouteCaptureSettings) -> dict[str, object]:
    return {
        "schema_version": 1,
        "target_window_title": settings.target_window_title,
        "capture_backend": settings.capture_backend,
        "emergency_virtual_key": settings.emergency_virtual_key,
        "max_key_hold_ms": settings.max_key_hold_ms,
        "route_capture_watchdog_ms": min(
            settings.max_key_hold_ms,
            ROUTE_CAPTURE_WATCHDOG_MS,
        ),
        "menu_profile_id": settings.menu_profile.profile_id,
        "menu_profile_sha256": settings.menu_profile.profile_sha256,
        "frame_size": list(settings.menu_profile.frame_size),
        "dataset_split": settings.dataset_split,
        "guard_interval_ms": settings.guard_interval_ms,
        "max_duration_seconds": float(settings.max_duration_seconds),
        "steps": [
            {
                "step_id": step.step_id,
                "keys": list(step.keys),
                "pulse_ms": step.pulse_ms,
                "mouse_dx": step.mouse_dx,
                "mouse_dy": step.mouse_dy,
                "settle_ms": step.settle_ms,
            }
            for step in settings.steps
        ],
    }


def route_capture_plan_sha256(settings: RouteCaptureSettings) -> str:
    payload = json.dumps(
        canonical_route_capture_plan(settings),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"delta-route-capture-plan-v1\0" + payload).hexdigest()
