"""受保护的 Windows 路线标定采集入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .artifact_io import write_atomic_json
from .config import CaptureRegion
from .events import JsonlEventWriter, RuntimeEvent
from .frames import CapturedFrame, FrameRecorder
from .menu_automation import MenuScene, SceneDecisionReason, SceneObservation
from .route_capture_config import (
    REQUIRED_FRAME_SIZE,
    RouteCaptureSettings,
    RouteCaptureStep,
    canonical_route_capture_plan,
    load_route_capture_settings,
    route_capture_plan_sha256,
)
from .runtime_events import persist_new_input_events


class RouteFrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


class RouteSceneObserver(Protocol):
    def observe(self, frame: CapturedFrame) -> SceneObservation: ...


class RouteInputActuator(Protocol):
    @property
    def events(self) -> tuple[object, ...]: ...

    @property
    def pressed_keys(self) -> frozenset[str]: ...

    def move_mouse_relative(self, dx: int, dy: int, *, now_ns: int) -> None: ...

    def key_down(self, key: str, *, now_ns: int) -> None: ...

    def key_up(
        self,
        key: str,
        *,
        now_ns: int,
        reason: str | None = None,
    ) -> None: ...

    def release_all(self, *, now_ns: int, reason: str) -> None: ...


class RouteWindowProbe(Protocol):
    def foreground_window_handle(self) -> int: ...

    def window_title(self, window_handle: int) -> str: ...


class RouteSafetyGuard(Protocol):
    def check(self) -> None: ...


class RouteCaptureStatus(StrEnum):
    DRY_RUN_VALIDATED = "dry_run_validated"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RouteCaptureResult:
    status: RouteCaptureStatus
    run_id: str
    plan_sha256: str
    hud_frame_count: int
    route_frame_count: int
    completed_step_count: int
    actual_input_event_count: int
    duration_ns: int


@dataclass(slots=True)
class RouteCaptureRuntime:
    source: RouteFrameSource
    observer: RouteSceneObserver
    dataset_recorder: FrameRecorder
    hud_recorder: FrameRecorder
    event_writer: JsonlEventWriter
    actuator: RouteInputActuator | None
    guard: RouteSafetyGuard
    target_window_handle: int
    capture_region: CaptureRegion
    artifact_root: Path
    run_id: str = "route-01"


def _release_and_persist_inputs(
    *,
    actuator: RouteInputActuator,
    persist_inputs: Callable[[], object],
    now_ns: int,
    reason: str,
) -> None:
    """释放失败也要持久化已经实际产生的部分释放事件。"""
    release_error: BaseException | None = None
    try:
        actuator.release_all(now_ns=now_ns, reason=reason)
    except BaseException as error:
        release_error = error
    try:
        persist_inputs()
    except BaseException as persist_error:
        if release_error is None:
            raise
        release_error.add_note(
            "路线采集释放后的输入事件持久化失败: "
            f"{type(persist_error).__name__}: {persist_error}"
        )
    if release_error is not None:
        raise release_error


class GuardedPulseExecutor:
    """在按键保持期间按固定小间隔持续复核所有安全门。"""

    def __init__(
        self,
        *,
        actuator: RouteInputActuator,
        guard: RouteSafetyGuard,
        guard_interval_ms: int,
        maximum_frame_age_ms: int,
        persist_inputs: Callable[[], object],
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._actuator = actuator
        self._guard = guard
        self._guard_interval_ns = guard_interval_ms * 1_000_000
        self._maximum_frame_age_ns = maximum_frame_age_ms * 1_000_000
        self._persist_inputs = persist_inputs
        self._clock_ns = clock_ns
        self._sleep_fn = sleep_fn

    def _check_authorization(self, authorized_at_ns: int) -> int:
        now_ns = self._clock_ns()
        if (
            type(now_ns) is not int
            or now_ns < 0
            or now_ns < authorized_at_ns
        ):
            raise RuntimeError("路线采集时钟或授权截图时间非法")
        if now_ns - authorized_at_ns > self._maximum_frame_age_ns:
            raise RuntimeError("路线采集授权截图已经过期")
        self._guard.check()
        return now_ns

    def execute(self, step: RouteCaptureStep, *, authorized_at_ns: int) -> None:
        now_ns = self._check_authorization(authorized_at_ns)
        pulse_deadline_ns = now_ns + step.pulse_ms * 1_000_000
        primary_error: BaseException | None = None
        try:
            if step.mouse_dx or step.mouse_dy:
                self._actuator.move_mouse_relative(
                    step.mouse_dx,
                    step.mouse_dy,
                    now_ns=now_ns,
                )
                self._persist_inputs()
            for key in step.keys:
                now_ns = self._check_authorization(authorized_at_ns)
                self._actuator.key_down(key, now_ns=now_ns)
                self._persist_inputs()
            while True:
                now_ns = self._check_authorization(authorized_at_ns)
                if now_ns >= pulse_deadline_ns:
                    break
                sleep_ns = min(self._guard_interval_ns, pulse_deadline_ns - now_ns)
                self._sleep_fn(sleep_ns / 1_000_000_000)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                _release_and_persist_inputs(
                    actuator=self._actuator,
                    persist_inputs=self._persist_inputs,
                    now_ns=self._clock_ns(),
                    reason=f"路线采集步骤结束: {step.step_id}",
                )
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "路线采集释放输入失败: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )


@dataclass(slots=True)
class _FrameCursor:
    previous_sequence: int = -1
    previous_captured_at_ns: int = -1

    def validate(
        self,
        frame: CapturedFrame,
        *,
        now_ns: int,
        maximum_frame_age_ms: int,
    ) -> None:
        if (
            frame.sequence <= self.previous_sequence
            or frame.captured_at_ns <= self.previous_captured_at_ns
        ):
            raise RuntimeError("路线采集截图序号和时间戳必须严格递增")
        if frame.captured_at_ns > now_ns:
            raise RuntimeError("路线采集截图时间戳晚于当前时钟")
        if now_ns - frame.captured_at_ns > maximum_frame_age_ms * 1_000_000:
            raise RuntimeError("路线采集截图已经过期")
        actual_size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if actual_size != REQUIRED_FRAME_SIZE:
            raise RuntimeError("路线采集截图不是 1920x1080")
        self.previous_sequence = frame.sequence
        self.previous_captured_at_ns = frame.captured_at_ns


def _observation_payload(observation: SceneObservation) -> dict[str, object]:
    return {
        "scene": str(observation.scene),
        "candidate_scene": (
            None if observation.candidate_scene is None else str(observation.candidate_scene)
        ),
        "accepted": observation.accepted,
        "reason": str(observation.reason),
        "confidence": observation.confidence,
        "runner_up_confidence": observation.runner_up_confidence,
        "page_template_id": observation.page_template_id,
    }


def _observe_in_match(
    observer: RouteSceneObserver,
    frame: CapturedFrame,
) -> SceneObservation:
    observation = observer.observe(frame)
    if (
        observation.frame_sequence != frame.sequence
        or observation.captured_at_ns != frame.captured_at_ns
    ):
        raise RuntimeError("路线采集页面观察结果与截图身份不一致")
    if observation.reason is SceneDecisionReason.FRAME_SIZE_MISMATCH:
        raise RuntimeError("路线采集截图与菜单 Profile 分辨率不一致")
    return observation


def _guarded_grab(
    runtime: RouteCaptureRuntime,
    *,
    cursor: _FrameCursor,
    maximum_frame_age_ms: int,
    clock_ns: Callable[[], int],
) -> CapturedFrame | None:
    runtime.guard.check()
    frame = runtime.source.grab()
    runtime.guard.check()
    if frame is not None:
        cursor.validate(
            frame,
            now_ns=clock_ns(),
            maximum_frame_age_ms=maximum_frame_age_ms,
        )
    return frame


def _guarded_wait(
    milliseconds: int,
    *,
    runtime: RouteCaptureRuntime,
    guard_interval_ms: int,
    deadline_ns: int,
    clock_ns: Callable[[], int],
    sleep_fn: Callable[[float], None],
) -> None:
    wait_deadline_ns = clock_ns() + milliseconds * 1_000_000
    while clock_ns() < wait_deadline_ns:
        now_ns = clock_ns()
        if now_ns >= deadline_ns:
            raise TimeoutError("路线采集超过最大运行时长")
        runtime.guard.check()
        remaining_ns = wait_deadline_ns - now_ns
        sleep_ns = min(guard_interval_ms * 1_000_000, remaining_ns)
        sleep_fn(sleep_ns / 1_000_000_000)
    runtime.guard.check()


def _frame_metadata(
    *,
    runtime: RouteCaptureRuntime,
    settings: RouteCaptureSettings,
    plan_sha256: str,
    step_index: int,
    step: RouteCaptureStep,
    phase: str,
    observation: SceneObservation,
    input_event_ids: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "run_id": runtime.run_id,
        "dataset_kind": "guarded-route-capture",
        "dataset_split": settings.dataset_split,
        "plan_sha256": plan_sha256,
        "route_capture": {
            "step_index": step_index,
            "step_id": step.step_id,
            "phase": phase,
            "window_handle": runtime.target_window_handle,
            "client_region": asdict(runtime.capture_region),
            "observation": _observation_payload(observation),
            "input_event_ids": list(input_event_ids),
        },
    }


def _write_dataset_run(
    runtime: RouteCaptureRuntime,
    settings: RouteCaptureSettings,
    *,
    frame_count: int,
    completed: bool,
) -> None:
    filename = "run.json" if completed else "partial-run.json"
    write_atomic_json(
        runtime.artifact_root / "dataset" / filename,
        {
            "run_id": runtime.run_id,
            "dataset_split": settings.dataset_split,
            "dataset_kind": "guarded-route-capture",
            "frame_count": frame_count,
            "resolution": list(REQUIRED_FRAME_SIZE),
            "complete": completed,
        },
    )


def run_route_capture_loop(
    *,
    settings: RouteCaptureSettings,
    runtime: RouteCaptureRuntime,
    armed: bool,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RouteCaptureResult:
    if type(armed) is not bool:
        raise ValueError("armed 必须是布尔值")
    if armed and runtime.actuator is None:
        raise ValueError("Armed 路线采集缺少输入执行器")
    if not armed and runtime.actuator is not None:
        raise ValueError("DryRun 路线采集不能持有输入执行器")
    plan_sha256 = route_capture_plan_sha256(settings)
    started_at_ns = clock_ns()
    deadline_ns = started_at_ns + round(settings.max_duration_seconds * 1_000_000_000)
    cursor = _FrameCursor()
    hud_frame_count = 0
    route_frame_count = 0
    completed_step_count = 0
    input_event_cursor = 0
    input_payloads_by_step: list[dict[str, object]] = []
    primary_error: BaseException | None = None
    success_result: RouteCaptureResult | None = None

    def now() -> int:
        value = clock_ns()
        if type(value) is not int or value < 0:
            raise RuntimeError("路线采集时钟必须是非负整数纳秒")
        if value >= deadline_ns:
            raise TimeoutError("路线采集超过最大运行时长")
        return value

    def persist_inputs() -> list[dict[str, object]]:
        nonlocal input_event_cursor
        if runtime.actuator is None:
            return []
        input_event_cursor, payloads = persist_new_input_events(
            actuator=runtime.actuator,
            input_event_cursor=input_event_cursor,
            recorder=runtime.dataset_recorder,
            event_writer=runtime.event_writer,
        )
        input_payloads_by_step.extend(payloads)
        return payloads

    def write_summary(status: str, error: BaseException | None) -> None:
        actual_input_count = 0 if runtime.actuator is None else len(runtime.actuator.events)
        pressed = [] if runtime.actuator is None else sorted(runtime.actuator.pressed_keys)
        write_atomic_json(
            runtime.artifact_root / "route-capture-summary.json",
            {
                "status": status,
                "run_id": runtime.run_id,
                "armed": armed,
                "plan_sha256": plan_sha256,
                "hud_frame_count": hud_frame_count,
                "route_frame_count": route_frame_count,
                "completed_step_count": completed_step_count,
                "actual_input_event_count": actual_input_count,
                "pressed_keys_final": pressed,
                "error": None
                if error is None
                else {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
            },
        )

    def write_failure_evidence(error: BaseException) -> None:
        evidence_errors: list[BaseException] = []
        complete_marker = runtime.artifact_root / "dataset" / "run.json"
        if complete_marker.exists():
            try:
                complete_marker.unlink()
            except BaseException as evidence_error:
                evidence_errors.append(evidence_error)
        if route_frame_count:
            try:
                _write_dataset_run(
                    runtime,
                    settings,
                    frame_count=route_frame_count,
                    completed=False,
                )
            except BaseException as evidence_error:
                evidence_errors.append(evidence_error)
        try:
            runtime.event_writer.write(
                RuntimeEvent(
                    event_type="runtime_error",
                    at_ns=max(0, clock_ns()),
                    payload={
                        "exception_type": type(error).__name__,
                        "error": str(error),
                        "completed_step_count": completed_step_count,
                        "route_frame_count": route_frame_count,
                    },
                )
            )
        except BaseException as evidence_error:
            evidence_errors.append(evidence_error)
        try:
            write_summary("failed", error)
        except BaseException as evidence_error:
            evidence_errors.append(evidence_error)
        for evidence_error in evidence_errors:
            error.add_note(
                "写入路线采集失败证据时再次失败: "
                f"{type(evidence_error).__name__}: {evidence_error}"
            )

    try:
        confirmation_count = 0
        confirmation_started_at_ns: int | None = None
        while confirmation_count < settings.menu_profile.confirmation_frames:
            current = now()
            frame = _guarded_grab(
                runtime,
                cursor=cursor,
                maximum_frame_age_ms=settings.menu_profile.maximum_frame_age_ms,
                clock_ns=clock_ns,
            )
            if frame is None:
                sleep_fn(settings.guard_interval_ms / 1_000)
                continue
            observation = _observe_in_match(runtime.observer, frame)
            runtime.hud_recorder.record(
                frame,
                metadata={
                    "run_id": runtime.run_id,
                    "plan_sha256": plan_sha256,
                    "hud_confirmation": _observation_payload(observation),
                },
            )
            hud_frame_count += 1
            runtime.event_writer.write(
                RuntimeEvent(
                    event_type="hud_observation",
                    at_ns=frame.captured_at_ns,
                    payload={
                        "frame_sequence": frame.sequence,
                        **_observation_payload(observation),
                    },
                )
            )
            if not observation.accepted or observation.scene is not MenuScene.IN_MATCH:
                confirmation_count = 0
                confirmation_started_at_ns = None
                continue
            if confirmation_started_at_ns is None or (
                frame.captured_at_ns - confirmation_started_at_ns
                > settings.menu_profile.maximum_confirmation_span_ms * 1_000_000
            ):
                confirmation_started_at_ns = frame.captured_at_ns
                confirmation_count = 1
            else:
                confirmation_count += 1
            if current >= deadline_ns:
                raise TimeoutError("路线采集 HUD 确认超时")

        if not armed:
            result = RouteCaptureResult(
                status=RouteCaptureStatus.DRY_RUN_VALIDATED,
                run_id=runtime.run_id,
                plan_sha256=plan_sha256,
                hud_frame_count=hud_frame_count,
                route_frame_count=0,
                completed_step_count=0,
                actual_input_event_count=0,
                duration_ns=max(0, clock_ns() - started_at_ns),
            )
            success_result = result
            return result

        if runtime.actuator is None:
            raise AssertionError("Armed 路线采集输入执行器丢失")
        executor = GuardedPulseExecutor(
            actuator=runtime.actuator,
            guard=runtime.guard,
            guard_interval_ms=settings.guard_interval_ms,
            maximum_frame_age_ms=settings.menu_profile.maximum_frame_age_ms,
            persist_inputs=persist_inputs,
            clock_ns=clock_ns,
            sleep_fn=sleep_fn,
        )
        for step_index, step in enumerate(settings.steps):
            input_payloads_by_step.clear()
            pre_frame = _guarded_grab(
                runtime,
                cursor=cursor,
                maximum_frame_age_ms=settings.menu_profile.maximum_frame_age_ms,
                clock_ns=clock_ns,
            )
            if pre_frame is None:
                raise RuntimeError(f"路线采集步骤 {step.step_id} 未取得 pre-frame")
            pre_observation = _observe_in_match(runtime.observer, pre_frame)
            if not pre_observation.accepted or pre_observation.scene is not MenuScene.IN_MATCH:
                raise RuntimeError(f"路线采集步骤 {step.step_id} 的 pre-frame 不是局内 HUD")
            runtime.dataset_recorder.record(
                pre_frame,
                metadata=_frame_metadata(
                    runtime=runtime,
                    settings=settings,
                    plan_sha256=plan_sha256,
                    step_index=step_index,
                    step=step,
                    phase="before",
                    observation=pre_observation,
                ),
            )
            route_frame_count += 1
            runtime.event_writer.write(
                RuntimeEvent(
                    event_type="step_pre_persisted",
                    at_ns=pre_frame.captured_at_ns,
                    payload={"step_index": step_index, "step_id": step.step_id},
                )
            )

            executor.execute(step, authorized_at_ns=pre_frame.captured_at_ns)
            _guarded_wait(
                step.settle_ms,
                runtime=runtime,
                guard_interval_ms=settings.guard_interval_ms,
                deadline_ns=deadline_ns,
                clock_ns=clock_ns,
                sleep_fn=sleep_fn,
            )
            post_frame = _guarded_grab(
                runtime,
                cursor=cursor,
                maximum_frame_age_ms=settings.menu_profile.maximum_frame_age_ms,
                clock_ns=clock_ns,
            )
            if post_frame is None:
                raise RuntimeError(f"路线采集步骤 {step.step_id} 未取得 post-frame")
            if runtime.actuator.events:
                last_input_at_ns = getattr(runtime.actuator.events[-1], "at_ns", None)
                if (
                    type(last_input_at_ns) is not int
                    or post_frame.captured_at_ns <= last_input_at_ns
                ):
                    raise RuntimeError("路线采集 post-frame 必须晚于最后一个实际输入事件")
            post_observation = _observe_in_match(runtime.observer, post_frame)
            if not post_observation.accepted or post_observation.scene is not MenuScene.IN_MATCH:
                raise RuntimeError(f"路线采集步骤 {step.step_id} 的 post-frame 不是局内 HUD")
            input_event_ids = tuple(
                payload["event_id"]
                for payload in input_payloads_by_step
                if isinstance(payload.get("event_id"), str)
            )
            runtime.dataset_recorder.record(
                post_frame,
                metadata=_frame_metadata(
                    runtime=runtime,
                    settings=settings,
                    plan_sha256=plan_sha256,
                    step_index=step_index,
                    step=step,
                    phase="after",
                    observation=post_observation,
                    input_event_ids=input_event_ids,
                ),
            )
            route_frame_count += 1
            completed_step_count += 1
            runtime.event_writer.write(
                RuntimeEvent(
                    event_type="step_completed",
                    at_ns=post_frame.captured_at_ns,
                    payload={
                        "step_index": step_index,
                        "step_id": step.step_id,
                        "input_event_ids": list(input_event_ids),
                    },
                )
            )

        result = RouteCaptureResult(
            status=RouteCaptureStatus.COMPLETED,
            run_id=runtime.run_id,
            plan_sha256=plan_sha256,
            hud_frame_count=hud_frame_count,
            route_frame_count=route_frame_count,
            completed_step_count=completed_step_count,
            actual_input_event_count=len(runtime.actuator.events),
            duration_ns=max(0, clock_ns() - started_at_ns),
        )
        success_result = result
        return result
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if runtime.actuator is not None:
            try:
                _release_and_persist_inputs(
                    actuator=runtime.actuator,
                    persist_inputs=persist_inputs,
                    now_ns=clock_ns(),
                    reason="路线采集最终清理",
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            runtime.source.close()
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        effective_error = primary_error
        if effective_error is None and cleanup_errors:
            effective_error = cleanup_errors.pop(0)
        if effective_error is not None:
            for cleanup_error in cleanup_errors:
                effective_error.add_note(
                    "路线采集异常清理失败: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            write_failure_evidence(effective_error)
            if primary_error is None:
                raise effective_error
        elif success_result is not None:
            try:
                write_summary(str(success_result.status), None)
                if success_result.status is RouteCaptureStatus.COMPLETED:
                    # 完整标记必须是所有清理与摘要成功后的最后一次提交。
                    _write_dataset_run(
                        runtime,
                        settings,
                        frame_count=route_frame_count,
                        completed=True,
                    )
            except BaseException as publication_error:
                write_failure_evidence(publication_error)
                raise


def build_windows_route_capture_runtime(
    settings: RouteCaptureSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str,
    **runtime_dependencies: object,
) -> RouteCaptureRuntime:
    # 延迟导入避免 Windows 装配模块反向引用运行时数据类型时形成初始化环。
    from .route_capture_windows import build_windows_route_capture_runtime as build

    return build(
        settings,
        artifacts=artifacts,
        armed=armed,
        run_id=run_id,
        **runtime_dependencies,
    )


def run_windows_route_capture(
    settings: RouteCaptureSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str,
) -> RouteCaptureResult:
    runtime = build_windows_route_capture_runtime(
        settings,
        artifacts=artifacts,
        armed=armed,
        run_id=run_id,
    )
    return run_route_capture_loop(settings=settings, runtime=runtime, armed=armed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows 1920x1080 受保护路线标定采集")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--confirm-armed", action="store_true")
    args = parser.parse_args(argv)
    if args.armed != args.confirm_armed:
        print(
            "路线采集 Armed 必须同时提供 --armed 和 --confirm-armed",
            file=sys.stderr,
        )
        return 1
    if args.validate_only and args.armed:
        print("--validate-only 不能与 Armed 同时使用", file=sys.stderr)
        return 1
    try:
        settings = load_route_capture_settings(args.config)
        plan = canonical_route_capture_plan(settings)
        plan_hash = route_capture_plan_sha256(settings)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "config": str(args.config),
                        "plan_sha256": plan_hash,
                        "plan": plan,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        artifacts = args.artifacts or (
            Path("artifacts/route-capture") / time.strftime("%Y%m%d-%H%M%S")
        )
        run_id = args.run_id or f"route-{uuid.uuid4().hex}"
        result = run_windows_route_capture(
            settings,
            artifacts=artifacts,
            armed=args.armed,
            run_id=run_id,
        )
    except Exception as error:
        print(f"路线采集启动或运行失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                **asdict(result),
                "status": str(result.status),
                "artifacts": str(artifacts),
                "armed": args.armed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
