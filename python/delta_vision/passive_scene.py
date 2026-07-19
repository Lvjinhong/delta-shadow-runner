"""只读观察赛后页面，连续确认已经返回大厅。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .artifact_io import write_atomic_json
from .capture import DxcamFrameSource, MssFrameSource
from .config import CaptureRegion
from .events import JsonlEventWriter, RuntimeEvent
from .frames import CapturedFrame, FrameRecorder
from .menu_automation import MenuScene, SceneDecisionReason, SceneObservation
from .menu_profile import LoadedMenuProfile
from .sample_frames import validate_run_id
from .win32_native import (
    Win32WindowProbe,
    find_window_handle,
    window_client_region_for_handle,
)


class PassiveFrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


class PassiveSceneObserver(Protocol):
    def observe(self, frame: CapturedFrame) -> SceneObservation: ...


class WindowProbe(Protocol):
    """被动回厅只需要窗口查询，禁止暴露输入方法。"""

    def foreground_window_handle(self) -> int: ...

    def window_title(self, window_handle: int) -> str: ...


class PassiveReturnStatus(StrEnum):
    BASE_CONFIRMED = "base_confirmed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class PassiveReturnPolicy:
    target_scene: MenuScene = MenuScene.BASE
    allowed_transient_scenes: frozenset[MenuScene] = frozenset(
        {MenuScene.IN_MATCH, MenuScene.DEATH_SUMMARY, MenuScene.POST_MATCH}
    )
    confirmation_frames: int = 3
    maximum_confirmation_span_ms: int = 500
    maximum_frame_age_ms: int = 500
    loop_interval_ms: int = 20
    max_duration_seconds: float = 180

    def __post_init__(self) -> None:
        if not isinstance(self.target_scene, MenuScene) or self.target_scene is MenuScene.UNKNOWN:
            raise ValueError("被动返回目标必须是非 UNKNOWN 的 MenuScene")
        if not isinstance(self.allowed_transient_scenes, frozenset) or any(
            not isinstance(scene, MenuScene) or scene in {MenuScene.UNKNOWN, self.target_scene}
            for scene in self.allowed_transient_scenes
        ):
            raise ValueError("允许的过渡页面必须是不含 UNKNOWN 和目标页的 MenuScene 集合")
        for value, field in (
            (self.confirmation_frames, "确认帧数"),
            (self.maximum_confirmation_span_ms, "最大确认跨度"),
            (self.maximum_frame_age_ms, "最大帧龄"),
            (self.loop_interval_ms, "循环间隔"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field}必须是正整数")
        if (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("最大观察时长必须是正有限数")


@dataclass(frozen=True, slots=True)
class PassiveReturnResult:
    status: PassiveReturnStatus
    terminal_scene: MenuScene
    seen_scenes: tuple[MenuScene, ...]
    frame_count: int
    duration_ns: int
    reason: str | None


def run_passive_return_loop(
    *,
    source: PassiveFrameSource,
    observer: PassiveSceneObserver,
    policy: PassiveReturnPolicy,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_observation: Callable[[CapturedFrame, SceneObservation], None] | None = None,
) -> PassiveReturnResult:
    """不发送输入，只在严格时序和帧龄约束下确认目标页面。"""

    frame_count = 0
    seen_scenes: list[MenuScene] = []
    previous_clock_ns = -1
    previous_sequence = -1
    previous_captured_at_ns = -1
    confirmation_count = 0
    confirmation_started_at_ns: int | None = None
    primary_error: BaseException | None = None

    def current_time_ns() -> int:
        nonlocal previous_clock_ns
        now_ns = clock_ns()
        if type(now_ns) is not int or now_ns < 0 or now_ns < previous_clock_ns:
            raise ValueError("被动返回观察时钟必须是单调非递减的非负整数纳秒")
        previous_clock_ns = now_ns
        return now_ns

    started_at_ns = 0

    def result(
        status: PassiveReturnStatus,
        terminal_scene: MenuScene,
        reason: str | None,
    ) -> PassiveReturnResult:
        return PassiveReturnResult(
            status=status,
            terminal_scene=terminal_scene,
            seen_scenes=tuple(seen_scenes),
            frame_count=frame_count,
            duration_ns=max(0, current_time_ns() - started_at_ns),
            reason=reason,
        )

    try:
        started_at_ns = current_time_ns()
        while True:
            now_ns = current_time_ns()
            if now_ns - started_at_ns >= policy.max_duration_seconds * 1_000_000_000:
                return result(
                    PassiveReturnStatus.STOPPED,
                    MenuScene.UNKNOWN,
                    "被动等待返回大厅超时",
                )

            frame = source.grab()
            if frame is None:
                sleep_fn(policy.loop_interval_ms / 1000)
                continue
            frame_count += 1
            now_ns = current_time_ns()
            if (
                frame.sequence <= previous_sequence
                or frame.captured_at_ns <= previous_captured_at_ns
            ):
                return result(
                    PassiveReturnStatus.STOPPED,
                    MenuScene.UNKNOWN,
                    "截图序号和时间戳必须严格递增",
                )
            previous_sequence = frame.sequence
            previous_captured_at_ns = frame.captured_at_ns
            if frame.captured_at_ns > now_ns:
                return result(
                    PassiveReturnStatus.STOPPED,
                    MenuScene.UNKNOWN,
                    "截图时间戳晚于当前时钟",
                )
            if now_ns - frame.captured_at_ns > policy.maximum_frame_age_ms * 1_000_000:
                return result(
                    PassiveReturnStatus.STOPPED,
                    MenuScene.UNKNOWN,
                    "截图已经过期，停止被动返回确认",
                )

            observation = observer.observe(frame)
            if (
                observation.frame_sequence != frame.sequence
                or observation.captured_at_ns != frame.captured_at_ns
            ):
                return result(
                    PassiveReturnStatus.STOPPED,
                    MenuScene.UNKNOWN,
                    "页面观察结果与截图身份不一致",
                )
            if on_observation is not None:
                on_observation(frame, observation)
            if observation.reason is SceneDecisionReason.FRAME_SIZE_MISMATCH:
                return result(
                    PassiveReturnStatus.STOPPED,
                    MenuScene.UNKNOWN,
                    "截图分辨率与页面 Profile 不一致",
                )
            if not observation.accepted:
                confirmation_count = 0
                confirmation_started_at_ns = None
                sleep_fn(policy.loop_interval_ms / 1000)
                continue

            seen_scenes.append(observation.scene)
            if observation.scene is policy.target_scene:
                if confirmation_started_at_ns is None:
                    confirmation_started_at_ns = frame.captured_at_ns
                    confirmation_count = 1
                elif (
                    frame.captured_at_ns - confirmation_started_at_ns
                    > policy.maximum_confirmation_span_ms * 1_000_000
                ):
                    confirmation_started_at_ns = frame.captured_at_ns
                    confirmation_count = 1
                else:
                    confirmation_count += 1
                if confirmation_count >= policy.confirmation_frames:
                    return result(
                        PassiveReturnStatus.BASE_CONFIRMED,
                        policy.target_scene,
                        None,
                    )
            elif observation.scene in policy.allowed_transient_scenes:
                confirmation_count = 0
                confirmation_started_at_ns = None
            else:
                return result(
                    PassiveReturnStatus.STOPPED,
                    observation.scene,
                    f"识别到未允许的页面: {observation.scene}",
                )
            sleep_fn(policy.loop_interval_ms / 1000)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            source.close()
        except BaseException as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"关闭被动返回截图源时出错: {type(close_error).__name__}: {close_error}"
            )


class _ForegroundGuardedSource:
    """每次截图前后复核绑定窗口和客户区，不尝试抢焦点。"""

    def __init__(
        self,
        source: PassiveFrameSource,
        *,
        window_probe: WindowProbe,
        target_window_handle: int,
        target_window_title: str,
        expected_region: CaptureRegion,
        region_resolver: Callable[[int], CaptureRegion],
    ) -> None:
        self._source = source
        self._window_probe = window_probe
        self._target_window_handle = target_window_handle
        self._target_window_title = target_window_title
        self._expected_region = expected_region
        self._region_resolver = region_resolver

    def _assert_binding(self, *, phase: str) -> None:
        foreground_handle = self._window_probe.foreground_window_handle()
        foreground_title = self._window_probe.window_title(foreground_handle)
        if (
            foreground_handle != self._target_window_handle
            or foreground_title != self._target_window_title
        ):
            raise RuntimeError(f"{phase}目标游戏不再是精确匹配的前台窗口")
        current_region = self._region_resolver(self._target_window_handle)
        if current_region != self._expected_region:
            raise RuntimeError(
                f"{phase}目标游戏客户区发生变化: "
                f"expected={self._expected_region}, actual={current_region}"
            )

    def grab(self) -> CapturedFrame | None:
        self._assert_binding(phase="截图前")
        frame = self._source.grab()
        self._assert_binding(phase="截图后")
        return frame

    def close(self) -> None:
        self._source.close()


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


def _result_payload(result: PassiveReturnResult) -> dict[str, object]:
    payload = asdict(result)
    payload["status"] = str(result.status)
    payload["terminal_scene"] = str(result.terminal_scene)
    payload["seen_scenes"] = [str(scene) for scene in result.seen_scenes]
    return payload


def run_windows_passive_return(
    *,
    menu_profile: LoadedMenuProfile,
    target_window_title: str,
    capture_backend: str,
    policy: PassiveReturnPolicy,
    artifacts: str | Path,
    run_id: str,
    region_resolver: Callable[[int], CaptureRegion] = window_client_region_for_handle,
    window_handle_resolver: Callable[[str], int] = find_window_handle,
    window_probe_factory: Callable[[], WindowProbe] = Win32WindowProbe,
    dxcam_factory: Callable[[CaptureRegion], PassiveFrameSource] = DxcamFrameSource,
    mss_factory: Callable[[CaptureRegion], PassiveFrameSource] = MssFrameSource,
) -> PassiveReturnResult:
    """Windows 只读入口：验证窗口、留存截图和事件，再连续确认大厅。"""

    parsed_run_id = validate_run_id(run_id)
    if capture_backend not in {"dxcam", "mss"}:
        raise ValueError('截图后端必须是 "dxcam" 或 "mss"')
    artifact_root = Path(artifacts)
    artifact_root.mkdir(parents=True, exist_ok=False)
    summary_path = artifact_root / "passive-return-summary.json"
    try:
        target_window_handle = window_handle_resolver(target_window_title)
        region = region_resolver(target_window_handle)
        if (region.width, region.height) != menu_profile.frame_size:
            raise ValueError(
                "菜单 Profile 分辨率与目标窗口客户区不一致: "
                f"expected={menu_profile.frame_size[0]}x{menu_profile.frame_size[1]}, "
                f"actual={region.width}x{region.height}"
            )
        window_probe = window_probe_factory()
        recorder = FrameRecorder(artifact_root / "replay")
        event_writer = JsonlEventWriter(
            artifact_root / "events.jsonl",
            run_id=parsed_run_id,
            truncate=True,
        )
        # 截图源最后创建，确保后续初始化不会在进入统一 close 路径前泄漏句柄。
        source_factory = dxcam_factory if capture_backend == "dxcam" else mss_factory
        source = _ForegroundGuardedSource(
            source_factory(region),
            window_probe=window_probe,
            target_window_handle=target_window_handle,
            target_window_title=target_window_title,
            expected_region=region,
            region_resolver=region_resolver,
        )

        def record(frame: CapturedFrame, observation: SceneObservation) -> None:
            payload = _observation_payload(observation)
            recorder.record(frame, metadata={"passive_return": payload})
            event_writer.write(
                RuntimeEvent(
                    event_type="passive_scene_observation",
                    at_ns=observation.captured_at_ns,
                    payload={"frame_sequence": frame.sequence, **payload},
                )
            )

        result = run_passive_return_loop(
            source=source,
            observer=menu_profile.observer,
            policy=policy,
            on_observation=record,
        )
        write_atomic_json(
            summary_path,
            {
                "schema_version": 1,
                "run_id": parsed_run_id,
                "status": str(result.status),
                "result": _result_payload(result),
            },
        )
        return result
    except BaseException as error:
        try:
            write_atomic_json(
                summary_path,
                {
                    "schema_version": 1,
                    "run_id": parsed_run_id,
                    "status": "failed",
                    "error": {
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    },
                },
            )
        except BaseException as summary_error:
            error.add_note(
                f"写入被动返回失败摘要时出错: {type(summary_error).__name__}: {summary_error}"
            )
        raise
