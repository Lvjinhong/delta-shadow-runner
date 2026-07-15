"""从 Windows 目标窗口只读采样人工路线截图。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .capture import DxcamFrameSource, MssFrameSource
from .config import CaptureRegion
from .frames import CapturedFrame, FrameRecorder
from .win32_native import window_client_region


class _FrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class FrameSamplingSchedule:
    """使用固定起点的节拍采样，落后时跳过过期槽位而不突发补帧。"""

    started_at_ns: int
    duration_seconds: float
    sample_fps: int
    _interval_ns: int = field(init=False, repr=False)
    _deadline_ns: int = field(init=False, repr=False)
    _next_sample_ns: int = field(init=False, repr=False)
    _previous_check_ns: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.started_at_ns) is not int or self.started_at_ns < 0:
            raise ValueError("采样开始时间必须是非负整数")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("采样持续时间必须是正有限数")
        if type(self.sample_fps) is not int or not 2 <= self.sample_fps <= 5:
            raise ValueError("人工路线采样频率必须是 2 到 5 FPS 的整数")
        self._interval_ns = round(1_000_000_000 / self.sample_fps)
        self._deadline_ns = self.started_at_ns + round(self.duration_seconds * 1_000_000_000)
        self._next_sample_ns = self.started_at_ns
        self._previous_check_ns = self.started_at_ns

    def consume(self, now_ns: int) -> bool:
        if type(now_ns) is not int or now_ns < self._previous_check_ns:
            raise ValueError("采样时钟必须单调递增")
        self._previous_check_ns = now_ns
        if now_ns >= self._deadline_ns or now_ns < self._next_sample_ns:
            return False
        elapsed_slots = (now_ns - self._next_sample_ns) // self._interval_ns + 1
        self._next_sample_ns += elapsed_slots * self._interval_ns
        return True

    def finished(self, now_ns: int) -> bool:
        return now_ns >= self._deadline_ns

    def sleep_seconds(self, now_ns: int) -> float:
        wake_at_ns = min(self._next_sample_ns, self._deadline_ns)
        return max(0, wake_at_ns - now_ns) / 1_000_000_000


@dataclass(frozen=True, slots=True)
class SamplingResult:
    run_id: str
    dataset_split: str
    window_title: str
    backend: str
    requested_duration_seconds: float
    measured_duration_seconds: float
    sample_fps: int
    frame_count: int
    no_frame_count: int
    resolution: tuple[int, int]


def _validate_run_id(run_id: object) -> str:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None
    ):
        raise ValueError("run_id 必须是 1 到 128 位字母、数字、点、下划线或短横线")
    return run_id


def _validate_dataset_split(dataset_split: object) -> str:
    if dataset_split not in {"calibration", "validation", "blind"}:
        raise ValueError('dataset_split 必须是 "calibration"、"validation" 或 "blind"')
    return dataset_split


def _write_result(path: Path, result: SamplingResult) -> None:
    payload = asdict(result)
    payload["resolution"] = list(result.resolution)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def sample_source(
    source: _FrameSource,
    *,
    output_directory: str | Path,
    run_id: str,
    dataset_split: str,
    window_title: str,
    backend: str,
    duration_seconds: float,
    sample_fps: int,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> SamplingResult:
    """按固定频率保存截图；此函数不导入或调用任何输入接口。"""

    try:
        parsed_run_id = _validate_run_id(run_id)
        parsed_dataset_split = _validate_dataset_split(dataset_split)
        if not isinstance(window_title, str) or not window_title.strip():
            raise ValueError("window_title 必须是非空字符串")
        if backend not in {"dxcam", "mss"}:
            raise ValueError('backend 只能是 "dxcam" 或 "mss"')
        output_root = Path(output_directory)
        if (output_root / "manifest.jsonl").exists() or (output_root / "run.json").exists():
            raise FileExistsError(f"采样目录已经存在运行数据: {output_root}")

        started_at_ns = clock_ns()
        schedule = FrameSamplingSchedule(
            started_at_ns=started_at_ns,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
        )
        recorder = FrameRecorder(output_root)
        frame_count = 0
        no_frame_count = 0
        resolution: tuple[int, int] | None = None
        while True:
            now_ns = clock_ns()
            if schedule.finished(now_ns):
                break
            if not schedule.consume(now_ns):
                sleep_fn(schedule.sleep_seconds(now_ns))
                continue
            frame = source.grab()
            if frame is None:
                no_frame_count += 1
                continue
            actual_resolution = (int(frame.image.shape[1]), int(frame.image.shape[0]))
            if resolution is None:
                resolution = actual_resolution
            elif actual_resolution != resolution:
                raise ValueError(
                    "采样期间发生分辨率漂移: "
                    f"expected={resolution[0]}x{resolution[1]}, "
                    f"actual={actual_resolution[0]}x{actual_resolution[1]}"
                )
            recorder.record(
                frame,
                metadata={
                    "run_id": parsed_run_id,
                    "dataset_kind": "manual-game-route",
                    "dataset_split": parsed_dataset_split,
                    "window_title": window_title,
                    "backend": backend,
                    "sample_fps": sample_fps,
                },
            )
            frame_count += 1
        ended_at_ns = clock_ns()
        if frame_count == 0 or resolution is None:
            raise RuntimeError("采样期间没有获得任何可用截图")
        result = SamplingResult(
            run_id=parsed_run_id,
            dataset_split=parsed_dataset_split,
            window_title=window_title,
            backend=backend,
            requested_duration_seconds=float(duration_seconds),
            measured_duration_seconds=max(0, ended_at_ns - started_at_ns) / 1_000_000_000,
            sample_fps=sample_fps,
            frame_count=frame_count,
            no_frame_count=no_frame_count,
            resolution=resolution,
        )
        _write_result(output_root / "run.json", result)
        return result
    finally:
        source.close()


def run_windows_sampling(
    *,
    window_title: str,
    backend: str,
    output_directory: str | Path,
    run_id: str,
    dataset_split: str,
    duration_seconds: float,
    sample_fps: int,
    start_delay_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    region_resolver: Callable[[str], CaptureRegion] = window_client_region,
    dxcam_factory: Callable[[CaptureRegion], _FrameSource] = DxcamFrameSource,
    mss_factory: Callable[[CaptureRegion], _FrameSource] = MssFrameSource,
) -> SamplingResult:
    if (
        isinstance(start_delay_seconds, bool)
        or not isinstance(start_delay_seconds, (int, float))
        or not math.isfinite(start_delay_seconds)
        or not 0 <= start_delay_seconds <= 300
    ):
        raise ValueError("开始倒计时必须是 0 到 300 秒的有限数")
    if start_delay_seconds:
        sleep_fn(start_delay_seconds)
    region = region_resolver(window_title)
    if backend == "dxcam":
        source = dxcam_factory(region)
    elif backend == "mss":
        source = mss_factory(region)
    else:
        raise ValueError('backend 只能是 "dxcam" 或 "mss"')
    return sample_source(
        source,
        output_directory=output_directory,
        run_id=run_id,
        dataset_split=dataset_split,
        window_title=window_title,
        backend=backend,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读采样 Windows 游戏窗口的人工路线截图")
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--backend", choices=("dxcam", "mss"), default="dxcam")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default=f"route-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument(
        "--split",
        dest="dataset_split",
        choices=("calibration", "validation", "blind"),
        required=True,
    )
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--fps", type=int, choices=range(2, 6), default=5)
    parser.add_argument("--start-delay", type=float, default=5)
    args = parser.parse_args(argv)
    try:
        result = run_windows_sampling(
            window_title=args.window_title,
            backend=args.backend,
            output_directory=args.output,
            run_id=args.run_id,
            dataset_split=args.dataset_split,
            duration_seconds=args.duration,
            sample_fps=args.fps,
            start_delay_seconds=args.start_delay,
        )
    except Exception as error:
        print(f"路线截图采样失败: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
