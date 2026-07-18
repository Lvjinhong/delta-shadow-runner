"""从菜单受保护进图，并在确认 HUD 后立即开始只读人工路线采样。"""

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

from .artifact_io import write_atomic_json
from .game_session import WindowsMenuSettings, run_windows_menu
from .menu_automation import MenuControllerStatus, MenuScene
from .menu_profile import load_menu_profile
from .menu_worker import MenuLoopResult
from .sample_frames import (
    FrameSamplingSchedule,
    SamplingResult,
    run_windows_sampling,
    validate_dataset_split,
    validate_run_id,
)


class SessionSamplingStatus(StrEnum):
    COMPLETED = "completed"
    MENU_STOPPED = "menu_stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SessionSamplingResult:
    status: SessionSamplingStatus
    menu: MenuLoopResult
    sampling: SamplingResult | None


def _menu_payload(result: MenuLoopResult) -> dict[str, object]:
    return {
        "status": str(result.status),
        "terminal_scene": str(result.terminal_scene),
        "frame_count": result.frame_count,
        "action_count": result.action_count,
        "duration_ns": result.duration_ns,
        "reason": result.reason,
    }


def _sampling_payload(result: SamplingResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    payload = asdict(result)
    payload["resolution"] = list(result.resolution)
    return payload


def _failure_payload(
    *,
    run_id: str,
    failed_phase: str,
    error: BaseException,
    menu_result: MenuLoopResult | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": str(SessionSamplingStatus.FAILED),
        "failed_phase": failed_phase,
        "menu": None if menu_result is None else _menu_payload(menu_result),
        "sampling": None,
        "error": {
            "exception_type": type(error).__name__,
            "message": str(error),
        },
    }


def _write_failure_summary(
    *,
    path: Path,
    run_id: str,
    failed_phase: str,
    error: BaseException,
    menu_result: MenuLoopResult | None,
) -> None:
    try:
        write_atomic_json(
            path,
            _failure_payload(
                run_id=run_id,
                failed_phase=failed_phase,
                error=error,
                menu_result=menu_result,
            ),
        )
    except BaseException as summary_error:
        error.add_note(
            "写入进图采样失败摘要时出错: "
            f"{type(summary_error).__name__}: {summary_error}"
        )


def _validate_sampling_policy(
    *,
    run_id: object,
    dataset_split: object,
    duration_seconds: object,
    sample_fps: object,
) -> tuple[str, str]:
    parsed_run_id = validate_run_id(run_id)
    parsed_split = validate_dataset_split(dataset_split)
    # 复用采样器的参数约束，但不创建目录、截图源或输入设备。
    FrameSamplingSchedule(
        started_at_ns=0,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
    )
    return parsed_run_id, parsed_split


def run_session_sampling(
    settings: WindowsMenuSettings,
    *,
    artifacts: str | Path,
    run_id: str,
    dataset_split: str,
    duration_seconds: float,
    sample_fps: int,
    menu_runner: Callable[..., MenuLoopResult] = run_windows_menu,
    sampling_runner: Callable[..., SamplingResult] = run_windows_sampling,
) -> SessionSamplingResult:
    """菜单完成前不采样；菜单完成后不再发送输入，只启动截图采样器。"""

    parsed_run_id, parsed_split = _validate_sampling_policy(
        run_id=run_id,
        dataset_split=dataset_split,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
    )
    artifact_root = Path(artifacts)
    try:
        artifact_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"运行目录已经存在，拒绝覆盖: {artifact_root}") from error
    summary_path = artifact_root / "session-sample-summary.json"
    try:
        menu_result = menu_runner(
            settings=settings,
            artifacts=artifact_root / "menu",
            armed=True,
            run_id=parsed_run_id,
        )
    except BaseException as error:
        _write_failure_summary(
            path=summary_path,
            run_id=parsed_run_id,
            failed_phase="menu",
            error=error,
            menu_result=None,
        )
        raise

    if menu_result.status is not MenuControllerStatus.COMPLETED:
        result = SessionSamplingResult(
            status=SessionSamplingStatus.MENU_STOPPED,
            menu=menu_result,
            sampling=None,
        )
        write_atomic_json(
            summary_path,
            {
                "schema_version": 1,
                "run_id": parsed_run_id,
                "status": str(result.status),
                "menu": _menu_payload(menu_result),
                "sampling": None,
            },
        )
        return result

    if menu_result.terminal_scene is not MenuScene.IN_MATCH:
        error = RuntimeError("菜单报告完成，但终态不是 IN_MATCH，拒绝开始采样")
        _write_failure_summary(
            path=summary_path,
            run_id=parsed_run_id,
            failed_phase="menu_contract",
            error=error,
            menu_result=menu_result,
        )
        raise error

    try:
        sampling_result = sampling_runner(
            window_title=settings.target_window_title,
            backend=settings.capture_backend,
            output_directory=artifact_root / "dataset",
            run_id=parsed_run_id,
            dataset_split=parsed_split,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
            start_delay_seconds=0,
        )
    except BaseException as error:
        _write_failure_summary(
            path=summary_path,
            run_id=parsed_run_id,
            failed_phase="sampling",
            error=error,
            menu_result=menu_result,
        )
        raise

    result = SessionSamplingResult(
        status=SessionSamplingStatus.COMPLETED,
        menu=menu_result,
        sampling=sampling_result,
    )
    write_atomic_json(
        summary_path,
        {
            "schema_version": 1,
            "run_id": parsed_run_id,
            "status": str(result.status),
            "menu": _menu_payload(menu_result),
            "sampling": _sampling_payload(sampling_result),
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从菜单受保护进图，并在确认 HUD 后立即只读采样人工路线"
    )
    parser.add_argument("--menu-profile", type=Path, required=True)
    parser.add_argument("--window-title", default="三角洲行动")
    parser.add_argument("--backend", choices=("dxcam", "mss"), default="dxcam")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run-id", default=f"route-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument(
        "--split",
        dest="dataset_split",
        choices=("calibration", "validation", "blind"),
        required=True,
    )
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--fps", type=int, choices=range(2, 6), default=5)
    parser.add_argument("--menu-loop-interval-ms", type=int, default=20)
    parser.add_argument("--menu-max-duration", type=float, default=180)
    parser.add_argument("--emergency-virtual-key", type=int, default=123)
    parser.add_argument("--max-key-hold-ms", type=int, default=250)
    parser.add_argument("--armed", action="store_true")
    args = parser.parse_args(argv)
    if not args.armed:
        print("SessionSample 必须显式传入 --armed；F12 是急停键。", file=sys.stderr)
        return 1
    try:
        settings = WindowsMenuSettings(
            target_window_title=args.window_title,
            capture_backend=args.backend,
            emergency_virtual_key=args.emergency_virtual_key,
            max_key_hold_ms=args.max_key_hold_ms,
            menu_profile=load_menu_profile(args.menu_profile),
            loop_interval_ms=args.menu_loop_interval_ms,
            max_duration_seconds=args.menu_max_duration,
        )
        result = run_session_sampling(
            settings,
            artifacts=args.artifacts,
            run_id=args.run_id or uuid.uuid4().hex,
            dataset_split=args.dataset_split,
            duration_seconds=args.duration,
            sample_fps=args.fps,
        )
    except Exception as error:
        print(f"进图采样失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": str(result.status),
                "run_id": args.run_id,
                "artifacts": str(args.artifacts),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.status is SessionSamplingStatus.COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
