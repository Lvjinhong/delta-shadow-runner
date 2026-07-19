"""路线标定采集的 Windows 安全装配。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .capture import DxcamFrameSource, MssFrameSource
from .config import CaptureRegion
from .events import JsonlEventWriter
from .frames import FrameRecorder
from .route_capture import (
    RouteCaptureRuntime,
    RouteFrameSource,
    RouteInputActuator,
    RouteWindowProbe,
)
from .route_capture_config import (
    REQUIRED_FRAME_SIZE,
    ROUTE_CAPTURE_WATCHDOG_MS,
    RouteCaptureSettings,
)
from .safe_input import SafetyGate, Win32InputActuator
from .sample_frames import validate_run_id
from .win32_native import (
    Win32NativeGateway,
    Win32WindowProbe,
    find_window_handle,
    window_client_region_for_handle,
)
from .worker import SCAN_CODES


class ExactWindowGuard:
    """绑定唯一 HWND、完整标题和首次客户区；不尝试抢焦点。"""

    def __init__(
        self,
        *,
        target_window_title: str,
        target_window_handle: int,
        expected_region: CaptureRegion,
        region_resolver: Callable[[int], CaptureRegion],
        window_probe: RouteWindowProbe | None,
        input_gate: SafetyGate | None,
    ) -> None:
        self._target_window_title = target_window_title
        self._target_window_handle = target_window_handle
        self._expected_region = expected_region
        self._region_resolver = region_resolver
        self._window_probe = window_probe
        self._input_gate = input_gate
        if (window_probe is None) == (input_gate is None):
            raise ValueError("窗口守卫必须且只能配置一种前台检查器")

    def check(self) -> None:
        if self._input_gate is not None:
            self._input_gate.check()
        else:
            if self._window_probe is None:
                raise AssertionError("只读窗口守卫缺少 probe")
            foreground = self._window_probe.foreground_window_handle()
            title = self._window_probe.window_title(foreground)
            if (
                foreground != self._target_window_handle
                or title != self._target_window_title
            ):
                raise RuntimeError("目标游戏不再是精确匹配的前台窗口")
        current_region = self._region_resolver(self._target_window_handle)
        if current_region != self._expected_region:
            raise RuntimeError(
                "目标游戏客户区发生变化: "
                f"expected={self._expected_region}, actual={current_region}"
            )


def build_windows_route_capture_runtime(
    settings: RouteCaptureSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str,
    window_handle_resolver: Callable[[str], int] = find_window_handle,
    region_resolver: Callable[[int], CaptureRegion] = window_client_region_for_handle,
    window_probe_factory: Callable[[], RouteWindowProbe] = Win32WindowProbe,
    dxcam_factory: Callable[[CaptureRegion], RouteFrameSource] = DxcamFrameSource,
    mss_factory: Callable[[CaptureRegion], RouteFrameSource] = MssFrameSource,
    gateway_factory: Callable[[], Win32NativeGateway] = Win32NativeGateway,
) -> RouteCaptureRuntime:
    parsed_run_id = validate_run_id(run_id)
    if type(armed) is not bool:
        raise ValueError("armed 必须是布尔值")
    if armed and not settings.armed_ready:
        raise ValueError("路线采集 armed 被 route_capture.armed_ready=false 阻止")
    target_handle = window_handle_resolver(settings.target_window_title)
    region = region_resolver(target_handle)
    if (region.width, region.height) != REQUIRED_FRAME_SIZE:
        raise ValueError(
            "路线采集要求 1920x1080 客户区: "
            f"actual={region.width}x{region.height}"
        )

    artifact_root = Path(artifacts)
    artifact_root.mkdir(parents=True, exist_ok=False)
    source: RouteFrameSource | None = None
    try:
        if armed:
            gateway = gateway_factory()
            input_gate = SafetyGate(
                target_window_title=settings.target_window_title,
                target_window_handle=target_handle,
                emergency_virtual_key=settings.emergency_virtual_key,
                gateway=gateway,
            )
            guard = ExactWindowGuard(
                target_window_title=settings.target_window_title,
                target_window_handle=target_handle,
                expected_region=region,
                region_resolver=region_resolver,
                window_probe=None,
                input_gate=input_gate,
            )
            allowed_keys = {key for step in settings.steps for key in step.keys}
            actuator: RouteInputActuator | None = Win32InputActuator(
                scan_codes={key: SCAN_CODES[key] for key in allowed_keys},
                max_key_hold_ms=min(
                    settings.max_key_hold_ms,
                    ROUTE_CAPTURE_WATCHDOG_MS,
                ),
                gate=input_gate,
                gateway=gateway,
            )
        else:
            guard = ExactWindowGuard(
                target_window_title=settings.target_window_title,
                target_window_handle=target_handle,
                expected_region=region,
                region_resolver=region_resolver,
                window_probe=window_probe_factory(),
                input_gate=None,
            )
            actuator = None
        source_factory = (
            dxcam_factory if settings.capture_backend == "dxcam" else mss_factory
        )
        source = source_factory(region)
        return RouteCaptureRuntime(
            source=source,
            observer=settings.menu_profile.observer,
            dataset_recorder=FrameRecorder(artifact_root / "dataset"),
            hud_recorder=FrameRecorder(artifact_root / "hud"),
            event_writer=JsonlEventWriter(
                artifact_root / "audit.jsonl",
                run_id=parsed_run_id,
                truncate=True,
            ),
            actuator=actuator,
            guard=guard,
            target_window_handle=target_handle,
            capture_region=region,
            artifact_root=artifact_root,
            run_id=parsed_run_id,
        )
    except BaseException:
        if source is not None:
            source.close()
        raise
