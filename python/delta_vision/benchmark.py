"""Windows 截图后端的独立性能基准。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .capture import DxcamFrameSource, MssFrameSource
from .frames import CapturedFrame
from .win32_native import Win32NativeGateway, find_window_handle, window_client_region


class _FrameSource(Protocol):
    def grab(self) -> CapturedFrame | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CaptureMetrics:
    duration_seconds: float
    frame_count: int
    no_frame_count: int
    black_frame_count: int
    average_fps: float
    capture_latency_average_ms: float
    capture_latency_p95_ms: float
    capture_latency_max_ms: float
    initial_resolution: tuple[int, int] | None
    resolution_drift_count: int
    foreground_mismatch_count: int = 0
    schema_version: int = 2
    run_id: str | None = None
    started_at_ns: int | None = None
    ended_at_ns: int | None = None
    first_frame_sha256: str | None = None
    last_frame_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("schema_version 必须是整数 2")
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id.strip()
        ):
            raise ValueError("run_id 必须是非空字符串")
        numeric_fields = {
            "duration_seconds": self.duration_seconds,
            "average_fps": self.average_fps,
            "capture_latency_average_ms": self.capture_latency_average_ms,
            "capture_latency_p95_ms": self.capture_latency_p95_ms,
            "capture_latency_max_ms": self.capture_latency_max_ms,
        }
        for field_name, value in numeric_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field_name} 必须是非负有限数")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds 必须大于 0")

        count_fields = {
            "frame_count": self.frame_count,
            "no_frame_count": self.no_frame_count,
            "black_frame_count": self.black_frame_count,
            "resolution_drift_count": self.resolution_drift_count,
            "foreground_mismatch_count": self.foreground_mismatch_count,
        }
        for field_name, value in count_fields.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数")

        if self.black_frame_count > self.frame_count:
            raise ValueError("black_frame_count 不能超过 frame_count")
        if self.resolution_drift_count > self.frame_count:
            raise ValueError("resolution_drift_count 不能超过 frame_count")
        expected_fps = self.frame_count / self.duration_seconds
        if not math.isclose(
            self.average_fps,
            expected_fps,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("average_fps 与 frame_count / duration_seconds 不一致")
        if not (
            self.capture_latency_average_ms
            <= self.capture_latency_p95_ms
            <= self.capture_latency_max_ms
        ):
            raise ValueError(
                "截图延迟必须满足 capture_latency_average_ms <= "
                "capture_latency_p95_ms <= capture_latency_max_ms"
            )
        if self.initial_resolution is not None:
            if (
                not isinstance(self.initial_resolution, tuple)
                or len(self.initial_resolution) != 2
                or any(type(value) is not int or value <= 0 for value in self.initial_resolution)
            ):
                raise ValueError("initial_resolution 必须是两个正整数")
        timestamps = (self.started_at_ns, self.ended_at_ns)
        if any(value is not None for value in timestamps):
            if any(type(value) is not int or value < 0 for value in timestamps):
                raise ValueError("started_at_ns 和 ended_at_ns 必须同时是非负整数")
            assert self.started_at_ns is not None
            assert self.ended_at_ns is not None
            if self.ended_at_ns <= self.started_at_ns:
                raise ValueError("ended_at_ns 必须晚于 started_at_ns")
            measured_seconds = (self.ended_at_ns - self.started_at_ns) / 1_000_000_000
            if not math.isclose(
                self.duration_seconds,
                measured_seconds,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("duration_seconds 与单调时钟范围不一致")
        hashes = (self.first_frame_sha256, self.last_frame_sha256)
        if any(value is not None for value in hashes):
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            ):
                raise ValueError("首末帧 SHA-256 必须同时是 64 位小写十六进制字符串")


@dataclass(frozen=True, slots=True)
class CaptureBenchmark:
    metrics: CaptureMetrics
    first_frame: CapturedFrame | None
    last_frame: CapturedFrame | None


@dataclass(frozen=True, slots=True)
class CaptureGateResult:
    passed: bool
    failures: tuple[str, ...]
    schema_version: int = 1
    run_id: str | None = None


def evaluate_capture_metrics(
    metrics: CaptureMetrics,
    *,
    minimum_duration_seconds: float = 60,
    minimum_average_fps: float = 20,
    maximum_latency_p95_ms: float = 50,
) -> CaptureGateResult:
    failures: list[str] = []
    if metrics.duration_seconds < minimum_duration_seconds:
        failures.append("duration_below_minimum")
    if metrics.frame_count <= 0:
        failures.append("no_captured_frames")
    if metrics.initial_resolution is None:
        failures.append("missing_initial_resolution")
    if metrics.no_frame_count:
        failures.append("missing_frames")
    if metrics.black_frame_count:
        failures.append("black_frames")
    if metrics.average_fps < minimum_average_fps:
        failures.append("fps_below_minimum")
    if metrics.capture_latency_p95_ms > maximum_latency_p95_ms:
        failures.append("latency_p95_above_maximum")
    if metrics.resolution_drift_count:
        failures.append("resolution_drift")
    if metrics.foreground_mismatch_count:
        failures.append("foreground_window_mismatch")
    return CaptureGateResult(
        passed=not failures,
        failures=tuple(failures),
        run_id=metrics.run_id,
    )


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _is_near_black(image: np.ndarray) -> bool:
    """按有效亮像素比例识别黑帧，避免单个噪点绕过。"""

    # NumPy 的 axis max 会为 1440p 每帧分配并扫描一张完整中间图；OpenCV
    # 在原生代码中直接生成单通道掩码，保留相同判定语义且避免拖慢抓帧门禁。
    dark_mask = cv2.inRange(image, (0, 0, 0), (8, 8, 8))
    dark_pixel_ratio = float(cv2.countNonZero(dark_mask)) / dark_mask.size
    return dark_pixel_ratio >= 0.99


def benchmark_source(
    source: _FrameSource,
    *,
    duration_seconds: float,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    foreground_probe: Callable[[], bool] | None = None,
    run_id: str | None = None,
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
    black_frame_count = 0
    initial_resolution: tuple[int, int] | None = None
    resolution_drift_count = 0
    foreground_mismatch_count = 0
    first_frame: CapturedFrame | None = None
    last_frame: CapturedFrame | None = None
    try:
        while True:
            capture_started_ns = clock_ns()
            if capture_started_ns - started_at_ns >= duration_seconds * 1_000_000_000:
                break
            if foreground_probe is not None and not foreground_probe():
                foreground_mismatch_count += 1
            frame = source.grab()
            capture_finished_ns = clock_ns()
            latencies_ms.append(
                max(0, capture_finished_ns - capture_started_ns) / 1_000_000
            )
            if frame is None:
                no_frame_count += 1
                continue
            frame_count += 1
            if _is_near_black(frame.image):
                black_frame_count += 1
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
        black_frame_count=black_frame_count,
        average_fps=frame_count / measured_seconds if measured_seconds else 0,
        capture_latency_average_ms=average_latency_ms,
        capture_latency_p95_ms=_percentile_nearest_rank(latencies_ms, 0.95),
        capture_latency_max_ms=max(latencies_ms, default=0),
        initial_resolution=initial_resolution,
        resolution_drift_count=resolution_drift_count,
        foreground_mismatch_count=foreground_mismatch_count,
        run_id=run_id,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
    )
    return CaptureBenchmark(metrics=metrics, first_frame=first_frame, last_frame=last_frame)


def run_windows_benchmark(
    *,
    window_title: str,
    backend: str,
    duration_seconds: float,
    artifacts: str | Path,
    run_id: str,
) -> CaptureBenchmark:
    region = window_client_region(window_title)
    target_window_handle = find_window_handle(window_title)
    gateway = Win32NativeGateway()
    if backend == "dxcam":
        source: _FrameSource = DxcamFrameSource(region)
    elif backend == "mss":
        source = MssFrameSource(region)
    else:
        raise ValueError('backend 只能是 "dxcam" 或 "mss"')
    result = benchmark_source(
        source,
        duration_seconds=duration_seconds,
        foreground_probe=lambda: (
            gateway.foreground_window_handle() == target_window_handle
        ),
        run_id=run_id,
    )
    artifact_root = Path(artifacts)
    artifact_root.mkdir(parents=True, exist_ok=True)
    first_frame_sha256: str | None = None
    last_frame_sha256: str | None = None
    if result.first_frame is not None:
        first_frame_path = artifact_root / "first-frame.png"
        if not cv2.imwrite(str(first_frame_path), result.first_frame.image):
            raise OSError("写入首帧诊断图失败")
        first_frame_sha256 = hashlib.sha256(first_frame_path.read_bytes()).hexdigest()
    if result.last_frame is not None:
        last_frame_path = artifact_root / "last-frame.png"
        if not cv2.imwrite(str(last_frame_path), result.last_frame.image):
            raise OSError("写入末帧诊断图失败")
        last_frame_sha256 = hashlib.sha256(last_frame_path.read_bytes()).hexdigest()
    result = replace(
        result,
        metrics=replace(
            result.metrics,
            first_frame_sha256=first_frame_sha256,
            last_frame_sha256=last_frame_sha256,
        ),
    )
    metrics_path = artifact_root / "capture-metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(result.metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows 外部截图 60 秒性能基准")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--window-title")
    source.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=("dxcam", "mss"))
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        if args.config is not None:
            from .worker import load_worker_settings

            settings = load_worker_settings(args.config)
            window_title = settings.target_window_title
            backend = args.backend or settings.capture_backend
        else:
            window_title = args.window_title
            backend = args.backend or "dxcam"
        result = run_windows_benchmark(
            window_title=window_title,
            backend=backend,
            duration_seconds=args.duration,
            artifacts=args.artifacts,
            run_id=args.run_id or uuid.uuid4().hex,
        )
    except Exception as error:
        print(f"截图基准失败: {error}", file=sys.stderr)
        return 1
    gate = evaluate_capture_metrics(
        result.metrics,
        minimum_duration_seconds=args.duration,
    )
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "capture-gate.json").write_text(
        json.dumps(asdict(gate), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(result.metrics), ensure_ascii=False))
    if not gate.passed:
        print(
            f"截图基准未通过: {', '.join(gate.failures)}",
            file=sys.stderr,
        )
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
