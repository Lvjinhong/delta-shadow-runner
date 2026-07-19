"""仓库清理的截图循环、Windows 安全输入与会话证据。"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .artifact_io import write_atomic_json
from .capture import DxcamFrameSource, MssFrameSource
from .config import CaptureRegion
from .events import JsonlEventWriter, RuntimeEvent
from .external_loop import CleanupSessionResult
from .frames import CapturedFrame, FrameRecorder
from .menu_profile import LoadedMenuProfile
from .runtime_events import persist_new_input_events
from .safe_input import SafetyGate, Win32InputActuator
from .sample_frames import validate_run_id
from .warehouse_cleanup import (
    CleanupActionIntent,
    CleanupIntentKind,
    MenuProfileWarehouseObserver,
    WarehouseCleanupController,
    WarehouseCleanupPolicy,
    WarehouseCleanupSnapshot,
    WarehouseCleanupStatus,
    WarehouseObservation,
)
from .win32_native import (
    Win32NativeGateway,
    find_window_handle,
    window_client_region_for_handle,
)


class ExpiredCleanupActionError(RuntimeError):
    """仓库清理动作在 DryRun 记录前已经过期。"""


class DuplicateCleanupActionError(RuntimeError):
    """同一个仓库清理动作 ID 已经被消费。"""


class CleanupActionCleanupError(RuntimeError):
    """仓库 Armed 点击失败后，输入释放也失败。"""


class CleanupExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"


@dataclass(frozen=True, slots=True)
class CleanupExecutionRecord:
    status: CleanupExecutionStatus
    intent_id: str
    kind: str
    attempted_at_ns: int
    expires_at_ns: int
    local_position: tuple[int, int]
    screen_position: tuple[int, int]
    slot_index: int | None
    dry_run: bool


@dataclass(frozen=True, slots=True)
class WarehouseCleanupLoopResult:
    status: WarehouseCleanupStatus
    frame_count: int
    action_count: int
    duration_ns: int
    reason: str | None
    safe_box_count: int | None


class WarehouseFrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


class WarehouseObserver(Protocol):
    def observe(self, frame: CapturedFrame) -> WarehouseObservation: ...


class CleanupExecutor(Protocol):
    def execute(
        self,
        action: CleanupActionIntent,
        *,
        now_ns: int,
    ) -> CleanupExecutionRecord: ...


class ArmedCleanupInputActuator(Protocol):
    @property
    def events(self) -> tuple[object, ...]: ...

    def click_left_at(
        self,
        screen_x: int,
        screen_y: int,
        *,
        now_ns: int,
        expires_at_ns: int,
    ) -> None: ...

    def release_all(self, *, now_ns: int, reason: str) -> None: ...


def _validate_executor_time(*, now_ns: int, expires_at_ns: int) -> None:
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("仓库动作时钟必须是非负整数纳秒")
    if now_ns >= expires_at_ns:
        raise ExpiredCleanupActionError("仓库动作已经过期")


def _action_positions(
    action: CleanupActionIntent,
    *,
    capture_region: CaptureRegion,
) -> tuple[tuple[int, int], tuple[int, int]]:
    local_position = (round(action.position[0]), round(action.position[1]))
    if not (
        0 <= local_position[0] < capture_region.width
        and 0 <= local_position[1] < capture_region.height
    ):
        raise ValueError("仓库点击坐标不在采集区域内")
    return local_position, (
        capture_region.left + local_position[0],
        capture_region.top + local_position[1],
    )


class DryRunCleanupExecutor:
    """只记录清理点击的局部/屏幕坐标，绝不调用输入接口。"""

    def __init__(
        self,
        *,
        capture_region: CaptureRegion,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(capture_region, CaptureRegion):
            raise TypeError("capture_region 必须是 CaptureRegion")
        self._capture_region = capture_region
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._consumed_intent_ids: set[str] = set()
        self._records: list[CleanupExecutionRecord] = []

    @property
    def records(self) -> tuple[CleanupExecutionRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def execute(
        self,
        action: CleanupActionIntent,
        *,
        now_ns: int,
    ) -> CleanupExecutionRecord:
        if not isinstance(action, CleanupActionIntent):
            raise TypeError("action 必须是 CleanupActionIntent")
        _validate_executor_time(now_ns=now_ns, expires_at_ns=action.expires_at_ns)
        local_position, screen_position = _action_positions(
            action,
            capture_region=self._capture_region,
        )
        with self._lock:
            dispatch_now_ns = max(now_ns, self._clock_ns())
            _validate_executor_time(
                now_ns=dispatch_now_ns,
                expires_at_ns=action.expires_at_ns,
            )
            if action.intent_id in self._consumed_intent_ids:
                raise DuplicateCleanupActionError(
                    f"仓库清理动作禁止重复执行: {action.intent_id}"
                )
            self._consumed_intent_ids.add(action.intent_id)
            record = CleanupExecutionRecord(
                status=CleanupExecutionStatus.SUCCEEDED,
                intent_id=action.intent_id,
                kind=str(action.kind),
                attempted_at_ns=dispatch_now_ns,
                expires_at_ns=action.expires_at_ns,
                local_position=local_position,
                screen_position=screen_position,
                slot_index=action.slot_index,
                dry_run=True,
            )
            self._records.append(record)
            return record


class ArmedCleanupExecutor:
    """只允许已验证的打开/返回点击，并复用 Win32 安全门与释放语义。"""

    def __init__(
        self,
        *,
        actuator: ArmedCleanupInputActuator,
        capture_region: CaptureRegion,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(capture_region, CaptureRegion):
            raise TypeError("capture_region 必须是 CaptureRegion")
        self._actuator = actuator
        self._capture_region = capture_region
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._consumed_intent_ids: set[str] = set()
        self._records: list[CleanupExecutionRecord] = []

    @property
    def records(self) -> tuple[CleanupExecutionRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def execute(
        self,
        action: CleanupActionIntent,
        *,
        now_ns: int,
    ) -> CleanupExecutionRecord:
        if not isinstance(action, CleanupActionIntent):
            raise TypeError("action 必须是 CleanupActionIntent")
        if action.kind is CleanupIntentKind.TRANSFER_SLOT:
            raise ValueError("仓库 Armed 尚未验证非空转移")
        if action.kind not in {
            CleanupIntentKind.OPEN_WAREHOUSE,
            CleanupIntentKind.RETURN_BASE,
        }:
            raise ValueError("仓库 Armed 动作类型无效")
        _validate_executor_time(now_ns=now_ns, expires_at_ns=action.expires_at_ns)
        local_position, screen_position = _action_positions(
            action,
            capture_region=self._capture_region,
        )
        with self._lock:
            dispatch_now_ns = max(now_ns, self._clock_ns())
            _validate_executor_time(
                now_ns=dispatch_now_ns,
                expires_at_ns=action.expires_at_ns,
            )
            if action.intent_id in self._consumed_intent_ids:
                raise DuplicateCleanupActionError(
                    f"仓库清理动作禁止重复执行: {action.intent_id}"
                )
            # 输入一旦开始就不允许自动重试，避免部分成功后再次点击。
            self._consumed_intent_ids.add(action.intent_id)
            try:
                self._actuator.click_left_at(
                    *screen_position,
                    now_ns=dispatch_now_ns,
                    expires_at_ns=action.expires_at_ns,
                )
            except BaseException as action_error:
                try:
                    self._actuator.release_all(
                        now_ns=dispatch_now_ns,
                        reason=f"仓库动作失败: {action.intent_id}",
                    )
                except BaseException as cleanup_error:
                    raise CleanupActionCleanupError(
                        f"动作失败: {action_error}; 清理失败: {cleanup_error}"
                    ) from cleanup_error
                raise
            record = CleanupExecutionRecord(
                status=CleanupExecutionStatus.SUCCEEDED,
                intent_id=action.intent_id,
                kind=str(action.kind),
                attempted_at_ns=dispatch_now_ns,
                expires_at_ns=action.expires_at_ns,
                local_position=local_position,
                screen_position=screen_position,
                slot_index=action.slot_index,
                dry_run=False,
            )
            self._records.append(record)
            return record


def _execution_payload(
    record: CleanupExecutionRecord | None,
) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "status": str(record.status),
        "intent_id": record.intent_id,
        "kind": record.kind,
        "attempted_at_ns": record.attempted_at_ns,
        "expires_at_ns": record.expires_at_ns,
        "local_position": list(record.local_position),
        "screen_position": list(record.screen_position),
        "slot_index": record.slot_index,
        "dry_run": record.dry_run,
    }


def _snapshot_payload(
    snapshot: WarehouseCleanupSnapshot,
    *,
    execution: CleanupExecutionRecord | None,
) -> dict[str, object]:
    return {
        "status": str(snapshot.status),
        "phase": str(snapshot.phase),
        "reason": snapshot.reason,
        "observed_scene": str(snapshot.observed_scene),
        "safe_box_count": snapshot.safe_box_count,
        "execution": _execution_payload(execution),
    }


def run_warehouse_cleanup_loop(
    *,
    source: WarehouseFrameSource,
    observer: WarehouseObserver,
    controller: WarehouseCleanupController,
    executor: CleanupExecutor,
    recorder: FrameRecorder,
    event_writer: JsonlEventWriter,
    loop_interval_ms: int,
    max_duration_seconds: float,
    input_actuator: ArmedCleanupInputActuator | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> WarehouseCleanupLoopResult:
    """运行清理观察循环；executor 是否触碰输入由调用方显式决定。"""

    if type(loop_interval_ms) is not int or loop_interval_ms <= 0:
        raise ValueError("仓库循环间隔必须是正整数毫秒")
    if (
        isinstance(max_duration_seconds, bool)
        or not isinstance(max_duration_seconds, (int, float))
        or not math.isfinite(max_duration_seconds)
        or max_duration_seconds <= 0
    ):
        raise ValueError("仓库最大运行时长必须是正有限数")

    frame_count = 0
    action_count = 0
    snapshot: WarehouseCleanupSnapshot | None = None
    last_confirmed_safe_box_count: int | None = None
    last_valid_now_ns = 0
    source_close_attempted = False
    input_event_cursor = 0

    def current_time_ns() -> int:
        nonlocal last_valid_now_ns
        now_ns = clock_ns()
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("仓库 Worker 时钟必须是非负整数纳秒")
        last_valid_now_ns = now_ns
        return now_ns

    try:
        started_at_ns = current_time_ns()
        while True:
            observed_safe_box_count = None
            now_ns = current_time_ns()
            frame = None
            execution = None
            if now_ns - started_at_ns >= max_duration_seconds * 1_000_000_000:
                snapshot = controller.stop("仓库 Worker 运行超时")
            else:
                frame = source.grab()
                now_ns = current_time_ns()
                if now_ns - started_at_ns >= max_duration_seconds * 1_000_000_000:
                    snapshot = controller.stop("仓库 Worker 运行超时")
                elif frame is not None:
                    observation = observer.observe(frame)
                    snapshot = controller.step(observation, now_ns=now_ns)
                    observed_safe_box_count = snapshot.safe_box_count
                    if snapshot.action is not None:
                        execution = executor.execute(snapshot.action, now_ns=now_ns)
                        action_count += 1

            if input_actuator is not None:
                input_event_cursor, _ = persist_new_input_events(
                    actuator=input_actuator,
                    input_event_cursor=input_event_cursor,
                    recorder=recorder,
                    event_writer=event_writer,
                )

            if observed_safe_box_count is not None:
                last_confirmed_safe_box_count = observed_safe_box_count

            if frame is not None and snapshot is not None:
                payload = _snapshot_payload(snapshot, execution=execution)
                recorder.record(frame, metadata={"warehouse_cleanup": payload})
                event_writer.write(
                    RuntimeEvent(
                        event_type="warehouse_frame",
                        at_ns=now_ns,
                        payload=payload,
                    )
                )
                frame_count += 1
            if snapshot is not None and snapshot.status in {
                WarehouseCleanupStatus.COMPLETED,
                WarehouseCleanupStatus.STOPPED,
            }:
                break
            sleep_fn(loop_interval_ms / 1_000)

        ended_at_ns = current_time_ns()
        event_writer.write(
            RuntimeEvent(
                event_type="warehouse_terminal",
                at_ns=ended_at_ns,
                payload=_snapshot_payload(snapshot, execution=None),
            )
        )
        result = WarehouseCleanupLoopResult(
            status=snapshot.status,
            frame_count=frame_count,
            action_count=action_count,
            duration_ns=max(0, ended_at_ns - started_at_ns),
            reason=snapshot.reason,
            safe_box_count=last_confirmed_safe_box_count,
        )
        # close 可能不是幂等操作；调用前先记账，异常清理阶段不得再次关闭。
        source_close_attempted = True
        source.close()
        return result
    except BaseException as error:
        cleanup_errors: list[str] = []
        try:
            controller.stop("仓库 Worker 异常，执行安全停止")
        except BaseException as cleanup_error:
            cleanup_errors.append(f"controller.stop: {cleanup_error}")
        if input_actuator is not None:
            try:
                persist_new_input_events(
                    actuator=input_actuator,
                    input_event_cursor=input_event_cursor,
                    recorder=recorder,
                    event_writer=event_writer,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(f"persist_input_events: {cleanup_error}")
        if not source_close_attempted:
            try:
                source_close_attempted = True
                source.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(f"source.close: {cleanup_error}")
        try:
            event_writer.write(
                RuntimeEvent(
                    event_type="runtime_error",
                    at_ns=last_valid_now_ns,
                    payload={
                        "exception_type": type(error).__name__,
                        "error": str(error),
                        "cleanup_errors": cleanup_errors,
                    },
                )
            )
        except BaseException as cleanup_error:
            cleanup_errors.append(f"event_writer.write: {cleanup_error}")
        for cleanup_error in cleanup_errors:
            error.add_note(f"仓库异常清理阶段失败: {cleanup_error}")
        raise


@dataclass(frozen=True, slots=True)
class WindowsWarehouseCleanupSettings:
    """Windows 仓库清理的独立授权和固定运行参数。"""

    target_window_title: str
    capture_backend: str
    emergency_virtual_key: int
    max_key_hold_ms: int
    menu_profile: LoadedMenuProfile
    loop_interval_ms: int
    max_duration_seconds: float
    armed_ready: bool = False
    policy: WarehouseCleanupPolicy = field(default_factory=WarehouseCleanupPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.target_window_title, str) or not self.target_window_title:
            raise ValueError("目标窗口标题必须是非空字符串")
        if self.capture_backend not in {"dxcam", "mss"}:
            raise ValueError('截图后端必须是 "dxcam" 或 "mss"')
        if type(self.emergency_virtual_key) is not int or self.emergency_virtual_key != 123:
            raise ValueError("仓库清理急停键必须固定为 F12（virtual key 123）")
        if type(self.max_key_hold_ms) is not int or self.max_key_hold_ms <= 0:
            raise ValueError("最大按键保持时间必须是正整数毫秒")
        if type(self.loop_interval_ms) is not int or self.loop_interval_ms <= 0:
            raise ValueError("仓库循环间隔必须是正整数毫秒")
        if (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("仓库最大运行时长必须是正有限数")
        if type(self.armed_ready) is not bool:
            raise ValueError("仓库 armed_ready 必须是布尔值")
        if not isinstance(self.policy, WarehouseCleanupPolicy):
            raise TypeError("仓库 policy 必须是 WarehouseCleanupPolicy")
        if self.policy.nonempty_transfer_verified:
            raise ValueError("空保险箱 Profile 不能启用非空转移")
        MenuProfileWarehouseObserver(menu_profile=self.menu_profile)


@dataclass(frozen=True, slots=True)
class WindowsWarehouseCleanupRuntime:
    source: WarehouseFrameSource
    observer: MenuProfileWarehouseObserver
    controller: WarehouseCleanupController
    executor: DryRunCleanupExecutor | ArmedCleanupExecutor
    actuator: Win32InputActuator | None
    recorder: FrameRecorder
    event_writer: JsonlEventWriter
    target_window_handle: int


def build_windows_warehouse_cleanup_runtime(
    settings: WindowsWarehouseCleanupSettings,
    *,
    artifacts: str | Path,
    armed: bool = False,
    run_id: str | None = None,
    window_handle_resolver: Callable[[str], int] = find_window_handle,
    region_resolver: Callable[[int], CaptureRegion] = window_client_region_for_handle,
    dxcam_factory: Callable[[CaptureRegion], WarehouseFrameSource] = DxcamFrameSource,
    mss_factory: Callable[[CaptureRegion], WarehouseFrameSource] = MssFrameSource,
    gateway_factory: Callable[[], Win32NativeGateway] = Win32NativeGateway,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> WindowsWarehouseCleanupRuntime:
    """先校验授权、Profile 和客户区，再创建截图源与可选输入。"""

    if not isinstance(settings, WindowsWarehouseCleanupSettings):
        raise TypeError("settings 必须是 WindowsWarehouseCleanupSettings")
    if type(armed) is not bool:
        raise ValueError("armed 必须是布尔值")
    if armed and not settings.armed_ready:
        raise ValueError("仓库清理 armed 被配置 armed_ready=false 阻止")
    parsed_run_id = None if run_id is None else validate_run_id(run_id)
    observer = MenuProfileWarehouseObserver(menu_profile=settings.menu_profile)
    target_window_handle = window_handle_resolver(settings.target_window_title)
    region = region_resolver(target_window_handle)
    if (region.width, region.height) != settings.policy.expected_frame_size:
        raise ValueError(
            "仓库清理要求 1920x1080 客户区: "
            f"actual={region.width}x{region.height}"
        )
    source_factory = (
        dxcam_factory if settings.capture_backend == "dxcam" else mss_factory
    )
    source = source_factory(region)
    try:
        actuator: Win32InputActuator | None = None
        if armed:
            gateway = gateway_factory()
            actuator = Win32InputActuator(
                scan_codes={},
                max_key_hold_ms=settings.max_key_hold_ms,
                gate=SafetyGate(
                    target_window_title=settings.target_window_title,
                    target_window_handle=target_window_handle,
                    emergency_virtual_key=settings.emergency_virtual_key,
                    gateway=gateway,
                ),
                gateway=gateway,
                clock_ns=clock_ns,
            )
            executor: DryRunCleanupExecutor | ArmedCleanupExecutor = (
                ArmedCleanupExecutor(
                    actuator=actuator,
                    capture_region=region,
                    clock_ns=clock_ns,
                )
            )
        else:
            executor = DryRunCleanupExecutor(
                capture_region=region,
                clock_ns=clock_ns,
            )
        artifact_root = Path(artifacts)
        return WindowsWarehouseCleanupRuntime(
            source=source,
            observer=observer,
            controller=WarehouseCleanupController(settings.policy),
            executor=executor,
            actuator=actuator,
            recorder=FrameRecorder(artifact_root / "replay"),
            event_writer=JsonlEventWriter(
                artifact_root / "events.jsonl",
                run_id=parsed_run_id,
                truncate=True,
            ),
            target_window_handle=target_window_handle,
        )
    except BaseException as error:
        try:
            source.close()
        except BaseException as cleanup_error:
            error.add_note(f"初始化失败后关闭截图源失败: {cleanup_error}")
        raise


def _finalize_armed_runtime(
    runtime: WindowsWarehouseCleanupRuntime,
    *,
    now_ns: int,
) -> None:
    actuator = runtime.actuator
    if actuator is None:
        return
    release_error: BaseException | None = None
    try:
        actuator.release_all(now_ns=now_ns, reason="仓库 Worker 结束")
    except BaseException as error:
        release_error = error
    try:
        persist_new_input_events(
            actuator=actuator,
            input_event_cursor=0,
            recorder=runtime.recorder,
            event_writer=runtime.event_writer,
        )
    except BaseException as persist_error:
        if release_error is not None:
            release_error.add_note(f"持久化仓库输入事件失败: {persist_error}")
            raise release_error from persist_error
        raise
    if release_error is not None:
        raise release_error


def run_windows_warehouse_cleanup(
    settings: WindowsWarehouseCleanupSettings,
    *,
    artifacts: str | Path,
    armed: bool = False,
    run_id: str | None = None,
    window_handle_resolver: Callable[[str], int] = find_window_handle,
    region_resolver: Callable[[int], CaptureRegion] = window_client_region_for_handle,
    dxcam_factory: Callable[[CaptureRegion], WarehouseFrameSource] = DxcamFrameSource,
    mss_factory: Callable[[CaptureRegion], WarehouseFrameSource] = MssFrameSource,
    gateway_factory: Callable[[], Win32NativeGateway] = Win32NativeGateway,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> WarehouseCleanupLoopResult:
    runtime = build_windows_warehouse_cleanup_runtime(
        settings,
        artifacts=artifacts,
        armed=armed,
        run_id=run_id,
        window_handle_resolver=window_handle_resolver,
        region_resolver=region_resolver,
        dxcam_factory=dxcam_factory,
        mss_factory=mss_factory,
        gateway_factory=gateway_factory,
        clock_ns=clock_ns,
    )
    try:
        result = run_warehouse_cleanup_loop(
            source=runtime.source,
            observer=runtime.observer,
            controller=runtime.controller,
            executor=runtime.executor,
            recorder=runtime.recorder,
            event_writer=runtime.event_writer,
            loop_interval_ms=settings.loop_interval_ms,
            max_duration_seconds=settings.max_duration_seconds,
            input_actuator=runtime.actuator,
            clock_ns=clock_ns,
            sleep_fn=sleep_fn,
        )
    except BaseException as error:
        try:
            _finalize_armed_runtime(runtime, now_ns=clock_ns())
        except BaseException as cleanup_error:
            error.add_note(f"仓库 Armed 输入终结失败: {cleanup_error}")
        raise
    _finalize_armed_runtime(runtime, now_ns=clock_ns())
    return result


def _cleanup_summary(
    *,
    run_id: str,
    armed: bool,
    result: WarehouseCleanupLoopResult,
    completed: bool,
    reason: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "armed": armed,
        "dry_run": not armed,
        "status": str(result.status),
        "completed": completed,
        "frame_count": result.frame_count,
        "action_count": result.action_count,
        "duration_ns": result.duration_ns,
        "safe_box_count": result.safe_box_count,
        "reason": reason,
    }


class WindowsWarehouseCleanupSession:
    """把 Windows 仓库循环转换成外部循环要求的 CleanupSession。"""

    def __init__(
        self,
        *,
        settings: WindowsWarehouseCleanupSettings,
        runner: Callable[..., WarehouseCleanupLoopResult] = run_windows_warehouse_cleanup,
    ) -> None:
        if not isinstance(settings, WindowsWarehouseCleanupSettings):
            raise TypeError("settings 必须是 WindowsWarehouseCleanupSettings")
        if not callable(runner):
            raise TypeError("runner 必须可调用")
        self._settings = settings
        self._runner = runner

    def run(
        self,
        *,
        artifacts: Path,
        armed: bool,
        run_id: str,
    ) -> CleanupSessionResult:
        parsed_run_id = validate_run_id(run_id)
        if type(armed) is not bool:
            raise ValueError("armed 必须是布尔值")
        if armed and not self._settings.armed_ready:
            raise ValueError("仓库清理 armed 被配置 armed_ready=false 阻止")
        artifact_root = Path(artifacts)
        artifact_root.mkdir(parents=True, exist_ok=False)
        summary_path = artifact_root / "cleanup-summary.json"
        try:
            result = self._runner(
                self._settings,
                artifacts=artifact_root / "runtime",
                armed=armed,
                run_id=parsed_run_id,
            )
        except BaseException as error:
            try:
                write_atomic_json(
                    summary_path,
                    {
                        "schema_version": 1,
                        "run_id": parsed_run_id,
                        "armed": armed,
                        "dry_run": not armed,
                        "status": "failed",
                        "completed": False,
                        "error": {
                            "exception_type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                )
            except BaseException as summary_error:
                error.add_note(f"写入仓库失败摘要时出错: {summary_error}")
            raise
        if not isinstance(result, WarehouseCleanupLoopResult):
            raise TypeError("仓库 runner 必须返回 WarehouseCleanupLoopResult")
        completed = (
            result.status is WarehouseCleanupStatus.COMPLETED
            and result.safe_box_count == 0
        )
        reason = None if completed else (result.reason or "仓库终态未确认空保险箱")
        summary: Mapping[str, object] = _cleanup_summary(
            run_id=parsed_run_id,
            armed=armed,
            result=result,
            completed=completed,
            reason=reason,
        )
        write_atomic_json(summary_path, summary)
        return CleanupSessionResult(
            completed=completed,
            reason=reason,
            summary=summary,
        )
