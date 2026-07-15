"""Windows 截图后端的独立性能基准。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2

from .capture import DxcamFrameSource, MssFrameSource
from .frames import CapturedFrame
from .win32_native import window_client_region


class _FrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CaptureMetrics:
    duration_seconds: float
    frame_count: int
    no_frame_count: int
    average_fps: float
    capture_latency_average_ms: float
    capture_latency_p95_ms: float
    capture_latency_max_ms: float
    initial_resolution: tuple[int, int] | None
    resolution_drift_count: int


@dataclass(frozen=True, slots=True)
class CaptureBenchmark:
    metrics: CaptureMetrics
    first_frame: CapturedFrame | None
    last_frame: CapturedFrame | None


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def benchmark_source(
    source: _FrameSource,
    *,
    duration_seconds: float,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> CaptureBenchmark:
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("截图基准持续时间必须是正有限数")

    started_at_ns = clock_ns()
    latencies_ms: list[float] = []
    frame_count = 0
    no_frame_count = 0
    initial_resolution: tuple[int, int] | None = None
    resolution_drift_count = 0
    first_frame: CapturedFrame | None = None
    last_frame: CapturedFrame | None = None
    try:
        while True:
            capture_started_ns = clock_ns()
            if capture_started_ns - started_at_ns >= duration_seconds * 1_000_000_000:
                break
            frame = source.grab()
            capture_finished_ns = clock_ns()
            latencies_ms.append(
                max(0, capture_finished_ns - capture_started_ns) / 1_000_000
            )
            if frame is None:
                no_frame_count += 1
                continue
            frame_count += 1
            resolution = (int(frame.image.shape[1]), int(frame.image.shape[0]))
            if initial_resolution is None:
                initial_resolution = resolution
                first_frame = frame
            elif resolution != initial_resolution:
                resolution_drift_count += 1
            last_frame = frame
        ended_at_ns = clock_ns()
    finally:
        source.close()

    measured_seconds = max(0, ended_at_ns - started_at_ns) / 1_000_000_000
    average_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    metrics = CaptureMetrics(
        duration_seconds=measured_seconds,
        frame_count=frame_count,
        no_frame_count=no_frame_count,
        average_fps=frame_count / measured_seconds if measured_seconds else 0,
        capture_latency_average_ms=average_latency_ms,
        capture_latency_p95_ms=_percentile_nearest_rank(latencies_ms, 0.95),
        capture_latency_max_ms=max(latencies_ms, default=0),
        initial_resolution=initial_resolution,
        resolution_drift_count=resolution_drift_count,
    )
    return CaptureBenchmark(metrics=metrics, first_frame=first_frame, last_frame=last_frame)


def run_windows_benchmark(
    *,
    window_title: str,
    backend: str,
    duration_seconds: float,
    artifacts: str | Path,
) -> CaptureBenchmark:
    region = window_client_region(window_title)
    if backend == "dxcam":
        source: _FrameSource = DxcamFrameSource(region)
    elif backend == "mss":
        source = MssFrameSource(region)
    else:
        raise ValueError('backend 只能是 "dxcam" 或 "mss"')
    result = benchmark_source(source, duration_seconds=duration_seconds)
    artifact_root = Path(artifacts)
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_root / "capture-metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(result.metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result.first_frame is not None:
        if not cv2.imwrite(str(artifact_root / "first-frame.png"), result.first_frame.image):
            raise OSError("写入首帧诊断图失败")
    if result.last_frame is not None:
        if not cv2.imwrite(str(artifact_root / "last-frame.png"), result.last_frame.image):
            raise OSError("写入末帧诊断图失败")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows 外部截图 60 秒性能基准")
    parser.add_argument("--window-title", required=True)
    parser.add_argument("--backend", choices=("dxcam", "mss"), default="dxcam")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_windows_benchmark(
            window_title=args.window_title,
            backend=args.backend,
            duration_seconds=args.duration,
            artifacts=args.artifacts,
        )
    except Exception as error:
        print(f"截图基准失败: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result.metrics), ensure_ascii=False))
    passed = (
        result.metrics.average_fps >= 20
        and result.metrics.capture_latency_p95_ms <= 50
        and result.metrics.resolution_drift_count == 0
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
