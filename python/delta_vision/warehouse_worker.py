"""仓库清理的截图循环与无操作系统输入 DryRun 证据。"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .config import CaptureRegion
from .events import JsonlEventWriter, RuntimeEvent
from .frames import CapturedFrame, FrameRecorder
from .warehouse_cleanup import (
    CleanupActionIntent,
    WarehouseCleanupController,
    WarehouseCleanupSnapshot,
    WarehouseCleanupStatus,
    WarehouseObservation,
)


class ExpiredCleanupActionError(RuntimeError):
    """仓库清理动作在 DryRun 记录前已经过期。"""


class DuplicateCleanupActionError(RuntimeError):
    """同一个仓库清理动作 ID 已经被消费。"""


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

    @staticmethod
    def _validate_time(*, now_ns: int, expires_at_ns: int) -> None:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("仓库 DryRun 时钟必须是非负整数纳秒")
        if now_ns >= expires_at_ns:
            raise ExpiredCleanupActionError("仓库 DryRun 动作已经过期")

    def execute(
        self,
        action: CleanupActionIntent,
        *,
        now_ns: int,
    ) -> CleanupExecutionRecord:
        if not isinstance(action, CleanupActionIntent):
            raise TypeError("action 必须是 CleanupActionIntent")
        self._validate_time(now_ns=now_ns, expires_at_ns=action.expires_at_ns)
        local_x = round(action.position[0])
        local_y = round(action.position[1])
        if not (
            0 <= local_x < self._capture_region.width
            and 0 <= local_y < self._capture_region.height
        ):
            raise ValueError("仓库 DryRun 点击坐标不在采集区域内")
        screen_position = (
            self._capture_region.left + local_x,
            self._capture_region.top + local_y,
        )
        with self._lock:
            dispatch_now_ns = max(now_ns, self._clock_ns())
            self._validate_time(
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
                local_position=(local_x, local_y),
                screen_position=screen_position,
                slot_index=action.slot_index,
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
        "dry_run": True,
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
