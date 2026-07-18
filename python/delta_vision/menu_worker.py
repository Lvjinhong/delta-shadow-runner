"""菜单截图、确认、一次性输入与回放证据的闭环 Worker。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .events import JsonlEventWriter, RuntimeEvent
from .frames import CapturedFrame, FrameRecorder
from .menu_automation import (
    MenuControllerSnapshot,
    MenuControllerStatus,
    MenuScene,
    SceneObservation,
)
from .menu_runtime import MenuActionExecutor, MenuExecutionRecord
from .runtime_events import persist_new_input_events


class MenuFrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


class MenuSceneObserver(Protocol):
    def observe(self, frame: CapturedFrame) -> SceneObservation: ...


class MenuController(Protocol):
    def step(
        self,
        observation: SceneObservation,
        *,
        now_ns: int,
    ) -> MenuControllerSnapshot: ...

    def stop(self, reason: str) -> MenuControllerSnapshot: ...


class MenuActuator(Protocol):
    @property
    def pressed_keys(self) -> frozenset[str]: ...

    @property
    def events(self) -> tuple[object, ...]: ...

    def expire_overdue(self, *, now_ns: int) -> tuple[str, ...]: ...

    def release_all(self, *, now_ns: int, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class MenuLoopResult:
    status: MenuControllerStatus
    frame_count: int
    action_count: int
    duration_ns: int
    reason: str | None
    terminal_scene: MenuScene = MenuScene.UNKNOWN


def _execution_payload(record: MenuExecutionRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "status": str(record.status),
        "source": str(record.source),
        "expected_target": str(record.expected_target),
        "kind": str(record.kind),
        "attempted_at_ns": record.attempted_at_ns,
        "expires_at_ns": record.expires_at_ns,
        "local_position": list(record.local_position) if record.local_position else None,
        "screen_position": list(record.screen_position) if record.screen_position else None,
        "key": record.key,
        "error": record.error,
    }


def _snapshot_payload(
    snapshot: MenuControllerSnapshot,
    *,
    execution: MenuExecutionRecord | None,
    pressed_keys: frozenset[str],
) -> dict[str, object]:
    return {
        "status": str(snapshot.status),
        "transition_index": snapshot.transition_index,
        "reason": snapshot.reason,
        "observed_scene": str(snapshot.observed_scene),
        "observed_at_ns": snapshot.observed_at_ns,
        "execution": _execution_payload(execution),
        "pressed_keys": sorted(pressed_keys),
    }


def run_menu_control_loop(
    *,
    source: MenuFrameSource,
    observer: MenuSceneObserver,
    controller: MenuController,
    executor: MenuActionExecutor,
    actuator: MenuActuator,
    recorder: FrameRecorder,
    event_writer: JsonlEventWriter,
    loop_interval_ms: int,
    max_duration_seconds: float,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> MenuLoopResult:
    frame_count = 0
    action_count = 0
    input_event_cursor = 0
    snapshot: MenuControllerSnapshot | None = None
    last_valid_now_ns = 0

    def current_time_ns() -> int:
        nonlocal last_valid_now_ns
        now_ns = clock_ns()
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("菜单 Worker 时钟必须是非负整数纳秒")
        last_valid_now_ns = now_ns
        return now_ns

    try:
        if type(loop_interval_ms) is not int or loop_interval_ms <= 0:
            raise ValueError("菜单循环间隔必须是正整数毫秒")
        if (
            isinstance(max_duration_seconds, bool)
            or not isinstance(max_duration_seconds, (int, float))
            or not math.isfinite(max_duration_seconds)
            or max_duration_seconds <= 0
        ):
            raise ValueError("菜单最大运行时长必须是正有限数")

        started_at_ns = current_time_ns()
        while True:
            now_ns = current_time_ns()
            frame = None
            execution = None
            if now_ns - started_at_ns >= max_duration_seconds * 1_000_000_000:
                snapshot = controller.stop("菜单 Worker 运行超时")
            else:
                actuator.expire_overdue(now_ns=now_ns)
                frame = source.grab()
                now_ns = current_time_ns()
                actuator.expire_overdue(now_ns=now_ns)
                if now_ns - started_at_ns >= max_duration_seconds * 1_000_000_000:
                    snapshot = controller.stop("菜单 Worker 运行超时")
                elif frame is not None:
                    observation = observer.observe(frame)
                    snapshot = controller.step(observation, now_ns=now_ns)
                    if snapshot.action is not None:
                        execution = executor.execute(snapshot.action, now_ns=now_ns)
                        action_count += 1

            input_event_cursor, input_payloads = persist_new_input_events(
                actuator=actuator,
                input_event_cursor=input_event_cursor,
                recorder=recorder,
                event_writer=event_writer,
            )
            if frame is not None and snapshot is not None:
                payload = _snapshot_payload(
                    snapshot,
                    execution=execution,
                    pressed_keys=actuator.pressed_keys,
                )
                recorder.record(
                    frame,
                    metadata={"menu": payload, "input_events": input_payloads},
                )
                event_writer.write(
                    RuntimeEvent(event_type="menu_frame", at_ns=now_ns, payload=payload)
                )
                frame_count += 1
            if snapshot is not None and snapshot.status in {
                MenuControllerStatus.COMPLETED,
                MenuControllerStatus.STOPPED,
            }:
                break
            sleep_fn(loop_interval_ms / 1_000)

        ended_at_ns = current_time_ns()
        actuator.release_all(now_ns=ended_at_ns, reason="菜单 Worker 进入终态")
        input_event_cursor, _ = persist_new_input_events(
            actuator=actuator,
            input_event_cursor=input_event_cursor,
            recorder=recorder,
            event_writer=event_writer,
        )
        event_writer.write(
            RuntimeEvent(
                event_type="menu_terminal",
                at_ns=ended_at_ns,
                payload=_snapshot_payload(
                    snapshot,
                    execution=None,
                    pressed_keys=actuator.pressed_keys,
                ),
            )
        )
        result = MenuLoopResult(
            status=snapshot.status,
            frame_count=frame_count,
            action_count=action_count,
            duration_ns=max(0, ended_at_ns - started_at_ns),
            reason=snapshot.reason,
            terminal_scene=snapshot.observed_scene,
        )
        source.close()
    except BaseException as error:
        cleanup_errors: list[str] = []
        stopped_at_ns = last_valid_now_ns
        try:
            stopped_at_ns = current_time_ns()
        except BaseException as cleanup_error:
            cleanup_errors.append(
                f"clock_ns: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        try:
            controller.stop("菜单 Worker 异常，执行安全停止")
        except BaseException as cleanup_error:
            cleanup_errors.append(f"controller.stop: {cleanup_error}")
        try:
            actuator.release_all(now_ns=stopped_at_ns, reason="菜单 Worker 异常")
        except BaseException as cleanup_error:
            cleanup_errors.append(f"actuator.release_all: {cleanup_error}")
        try:
            persist_new_input_events(
                actuator=actuator,
                input_event_cursor=input_event_cursor,
                recorder=recorder,
                event_writer=event_writer,
            )
        except BaseException as cleanup_error:
            cleanup_errors.append(f"persist_input_events: {cleanup_error}")
        try:
            source.close()
        except BaseException as cleanup_error:
            cleanup_errors.append(f"source.close: {cleanup_error}")
        try:
            pressed_keys: list[str] | None = sorted(actuator.pressed_keys)
        except BaseException as cleanup_error:
            pressed_keys = None
            cleanup_errors.append(
                f"actuator.pressed_keys: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        error_payload: dict[str, object] = {
            "exception_type": type(error).__name__,
            "error": str(error),
            "pressed_keys": pressed_keys,
        }
        if cleanup_errors:
            error_payload["cleanup_errors"] = cleanup_errors
        try:
            event_writer.write(
                RuntimeEvent(
                    event_type="runtime_error",
                    at_ns=stopped_at_ns,
                    payload=error_payload,
                )
            )
        except BaseException as cleanup_error:
            cleanup_errors.append(f"event_writer.write: {cleanup_error}")
        for cleanup_error in cleanup_errors:
            error.add_note(f"异常清理阶段失败: {cleanup_error}")
        raise
    return result
