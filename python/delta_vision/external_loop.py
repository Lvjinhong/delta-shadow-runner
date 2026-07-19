"""大厅进图、局内运行、被动回厅和可选清理的外部循环。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .artifact_io import write_atomic_json
from .game_session import WindowsMenuSettings, run_windows_menu
from .menu_automation import (
    MenuActionKind,
    MenuControllerStatus,
    MenuScene,
)
from .menu_worker import MenuLoopResult
from .passive_scene import (
    PassiveReturnPolicy,
    PassiveReturnResult,
    PassiveReturnStatus,
    run_windows_passive_return,
)
from .sample_frames import validate_run_id
from .worker import ControlLoopResult, WorkerSettings, run_windows_worker


class ExternalLoopPhase(StrEnum):
    ENTERING = "entering"
    MATCH_RUNNING = "match_running"
    RETURN_OBSERVING = "return_observing"
    CLEANUP = "cleanup"


class ExternalLoopStatus(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CleanupSessionResult:
    completed: bool
    reason: str | None
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.completed) is not bool:
            raise ValueError("清理结果 completed 必须是布尔值")
        if self.completed and self.reason is not None:
            raise ValueError("清理完成时不能携带失败原因")
        if not self.completed and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("清理未完成时必须提供原因")
        if not isinstance(self.summary, Mapping):
            raise ValueError("清理结果 summary 必须是对象")
        object.__setattr__(self, "summary", dict(self.summary))


class CleanupSession(Protocol):
    def run(
        self,
        *,
        artifacts: Path,
        armed: bool,
        run_id: str,
    ) -> CleanupSessionResult: ...


@dataclass(frozen=True, slots=True)
class ExternalLoopSettings:
    entry: WindowsMenuSettings
    match: WorkerSettings
    return_policy: PassiveReturnPolicy
    cycle_limit: int = 1

    def __post_init__(self) -> None:
        if type(self.cycle_limit) is not int or self.cycle_limit <= 0:
            raise ValueError("外部循环次数必须是正整数")
        transitions = self.entry.menu_profile.transitions
        if (
            len(transitions) != 1
            or transitions[0].source is not MenuScene.BASE
            or transitions[0].action_kind is not MenuActionKind.CLICK
            or transitions[0].target is not MenuScene.IN_MATCH
        ):
            raise ValueError("外部循环进图必须且只能配置 BASE -> CLICK -> IN_MATCH")
        entry_frame_size = self.entry.menu_profile.frame_size
        match_frame_size = self.match.perception.frame_size
        if entry_frame_size != (1920, 1080) or match_frame_size != entry_frame_size:
            raise ValueError("外部循环要求大厅和局内 Profile 同为 1920x1080")
        if self.entry.target_window_title != self.match.target_window_title:
            raise ValueError("外部循环大厅和局内 Worker 的目标窗口标题必须完全一致")
        if self.entry.emergency_virtual_key != self.match.emergency_virtual_key:
            raise ValueError("外部循环大厅和局内 Worker 的急停键必须一致")
        if self.return_policy.target_scene is not MenuScene.BASE:
            raise ValueError("外部循环的被动返回目标必须是 BASE")


@dataclass(frozen=True, slots=True)
class ExternalLoopCycleResult:
    run_id: str
    entry: MenuLoopResult | None = None
    match: ControlLoopResult | None = None
    returned: PassiveReturnResult | None = None
    cleanup_started: bool = False
    cleanup_completed: bool = False
    cleanup_summary: Mapping[str, object] | None = None

    @property
    def cleanup_ran(self) -> bool:
        """兼容旧调用方；只有完整返回才视为清理已运行。"""

        return self.cleanup_completed


@dataclass(frozen=True, slots=True)
class ExternalLoopResult:
    status: ExternalLoopStatus
    completed_cycles: int
    cycles: tuple[ExternalLoopCycleResult, ...]
    stopped_phase: ExternalLoopPhase | None = None
    reason: str | None = None


def _entry_payload(result: MenuLoopResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "status": str(result.status),
        "terminal_scene": str(result.terminal_scene),
        "frame_count": result.frame_count,
        "action_count": result.action_count,
        "duration_ns": result.duration_ns,
        "reason": result.reason,
    }


def _match_payload(result: ControlLoopResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "status": str(result.status),
        "frame_count": result.frame_count,
        "duration_ns": result.duration_ns,
        "reason": result.reason,
    }


def _return_payload(result: PassiveReturnResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "status": str(result.status),
        "terminal_scene": str(result.terminal_scene),
        "seen_scenes": [str(scene) for scene in result.seen_scenes],
        "frame_count": result.frame_count,
        "duration_ns": result.duration_ns,
        "reason": result.reason,
    }


def _cycle_payload(cycle: ExternalLoopCycleResult) -> dict[str, object]:
    return {
        "run_id": cycle.run_id,
        "entry": _entry_payload(cycle.entry),
        "match": _match_payload(cycle.match),
        "return": _return_payload(cycle.returned),
        "cleanup_started": cycle.cleanup_started,
        "cleanup_completed": cycle.cleanup_completed,
        "cleanup_ran": cycle.cleanup_ran,
        "cleanup_summary": (None if cycle.cleanup_summary is None else dict(cycle.cleanup_summary)),
    }


def _summary_payload(
    *,
    run_id: str,
    armed: bool,
    status: ExternalLoopStatus,
    completed_cycles: int,
    cycles: list[ExternalLoopCycleResult],
    stopped_phase: ExternalLoopPhase | None = None,
    reason: str | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "armed": armed,
        "status": str(status),
        "completed_cycles": completed_cycles,
        "cycles": [_cycle_payload(cycle) for cycle in cycles],
        "stopped_phase": None if stopped_phase is None else str(stopped_phase),
        "reason": reason,
    }
    if error is not None:
        payload["failed_phase"] = None if stopped_phase is None else str(stopped_phase)
        payload["error"] = {
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    return payload


def _stopped_result(
    *,
    run_id: str,
    armed: bool,
    summary_path: Path,
    completed_cycles: int,
    cycles: list[ExternalLoopCycleResult],
    phase: ExternalLoopPhase,
    reason: str,
) -> ExternalLoopResult:
    write_atomic_json(
        summary_path,
        _summary_payload(
            run_id=run_id,
            armed=armed,
            status=ExternalLoopStatus.STOPPED,
            completed_cycles=completed_cycles,
            cycles=cycles,
            stopped_phase=phase,
            reason=reason,
        ),
    )
    return ExternalLoopResult(
        status=ExternalLoopStatus.STOPPED,
        completed_cycles=completed_cycles,
        cycles=tuple(cycles),
        stopped_phase=phase,
        reason=reason,
    )


def run_external_loop(
    settings: ExternalLoopSettings,
    *,
    artifacts: str | Path,
    armed: bool,
    run_id: str,
    entry_runner: Callable[..., MenuLoopResult] = run_windows_menu,
    match_runner: Callable[..., ControlLoopResult] = run_windows_worker,
    return_observer: Callable[..., PassiveReturnResult] = run_windows_passive_return,
    cleanup_session: CleanupSession | None = None,
    cleanup_hook: Callable[[ExternalLoopCycleResult], Any] | None = None,
) -> ExternalLoopResult:
    """每轮只有确认回到 BASE 后，才允许清理并开始下一轮。"""

    parsed_run_id = validate_run_id(run_id)
    if type(armed) is not bool:
        raise ValueError("armed 必须是布尔值")
    if armed and not settings.match.armed_ready:
        raise ValueError("外部循环 armed 被配置 armed_ready=false 阻止")
    if cleanup_session is not None and cleanup_hook is not None:
        raise ValueError("cleanup_session 与 cleanup_hook 不能同时配置")
    artifact_root = Path(artifacts)
    artifact_root.mkdir(parents=True, exist_ok=False)
    summary_path = artifact_root / "external-loop-summary.json"
    cycles: list[ExternalLoopCycleResult] = []
    completed_cycles = 0
    phase: ExternalLoopPhase | None = None
    current_cycle: ExternalLoopCycleResult | None = None

    try:
        for index in range(1, settings.cycle_limit + 1):
            cycle_run_id = validate_run_id(f"{parsed_run_id}-c{index:04d}")
            cycle_root = artifact_root / f"cycle-{index:04d}"
            current_cycle = ExternalLoopCycleResult(run_id=cycle_run_id)

            phase = ExternalLoopPhase.ENTERING
            entry = entry_runner(
                settings=settings.entry,
                artifacts=cycle_root / "entry",
                armed=armed,
                run_id=cycle_run_id,
            )
            current_cycle = replace(current_cycle, entry=entry)
            if entry.status is not MenuControllerStatus.COMPLETED:
                cycles.append(current_cycle)
                current_cycle = None
                return _stopped_result(
                    run_id=parsed_run_id,
                    armed=armed,
                    summary_path=summary_path,
                    completed_cycles=completed_cycles,
                    cycles=cycles,
                    phase=phase,
                    reason=entry.reason or "进图菜单流程未完成",
                )
            if entry.terminal_scene is not MenuScene.IN_MATCH:
                cycles.append(current_cycle)
                current_cycle = None
                return _stopped_result(
                    run_id=parsed_run_id,
                    armed=armed,
                    summary_path=summary_path,
                    completed_cycles=completed_cycles,
                    cycles=cycles,
                    phase=phase,
                    reason="进图菜单流程完成，但终态不是 IN_MATCH",
                )

            phase = ExternalLoopPhase.MATCH_RUNNING
            match = match_runner(
                settings.match,
                artifacts=cycle_root / "match",
                armed=armed,
                run_id=cycle_run_id,
            )
            current_cycle = replace(current_cycle, match=match)

            phase = ExternalLoopPhase.RETURN_OBSERVING
            returned = return_observer(
                menu_profile=settings.entry.menu_profile,
                target_window_title=settings.entry.target_window_title,
                capture_backend=settings.entry.capture_backend,
                policy=settings.return_policy,
                artifacts=cycle_root / "return",
                run_id=cycle_run_id,
            )
            current_cycle = replace(current_cycle, returned=returned)
            if (
                returned.status is not PassiveReturnStatus.BASE_CONFIRMED
                or returned.terminal_scene is not MenuScene.BASE
            ):
                cycles.append(current_cycle)
                current_cycle = None
                return _stopped_result(
                    run_id=parsed_run_id,
                    armed=armed,
                    summary_path=summary_path,
                    completed_cycles=completed_cycles,
                    cycles=cycles,
                    phase=phase,
                    reason=returned.reason or "未确认一致的 BASE 返回终态",
                )

            phase = ExternalLoopPhase.CLEANUP
            if cleanup_session is not None:
                current_cycle = replace(current_cycle, cleanup_started=True)
                cleanup = cleanup_session.run(
                    artifacts=cycle_root / "cleanup",
                    armed=armed,
                    run_id=cycle_run_id,
                )
                if not isinstance(cleanup, CleanupSessionResult):
                    raise TypeError("cleanup_session.run 必须返回 CleanupSessionResult")
                current_cycle = replace(
                    current_cycle,
                    cleanup_completed=cleanup.completed,
                    cleanup_summary=dict(cleanup.summary),
                )
                if not cleanup.completed:
                    cycles.append(current_cycle)
                    current_cycle = None
                    return _stopped_result(
                        run_id=parsed_run_id,
                        armed=armed,
                        summary_path=summary_path,
                        completed_cycles=completed_cycles,
                        cycles=cycles,
                        phase=phase,
                        reason=cleanup.reason or "仓库清理未完成",
                    )
            elif cleanup_hook is not None:
                current_cycle = replace(current_cycle, cleanup_started=True)
                cleanup_hook(current_cycle)
                current_cycle = replace(current_cycle, cleanup_completed=True)
            cycles.append(current_cycle)
            current_cycle = None
            completed_cycles += 1

        result = ExternalLoopResult(
            status=ExternalLoopStatus.COMPLETED,
            completed_cycles=completed_cycles,
            cycles=tuple(cycles),
        )
        write_atomic_json(
            summary_path,
            _summary_payload(
                run_id=parsed_run_id,
                armed=armed,
                status=result.status,
                completed_cycles=completed_cycles,
                cycles=cycles,
            ),
        )
        return result
    except BaseException as error:
        if current_cycle is not None:
            cycles.append(current_cycle)
        try:
            write_atomic_json(
                summary_path,
                _summary_payload(
                    run_id=parsed_run_id,
                    armed=armed,
                    status=ExternalLoopStatus.FAILED,
                    completed_cycles=completed_cycles,
                    cycles=cycles,
                    stopped_phase=phase,
                    reason=str(error),
                    error=error,
                ),
            )
        except BaseException as summary_error:
            error.add_note(
                f"写入外部循环失败摘要时出错: {type(summary_error).__name__}: {summary_error}"
            )
        raise
