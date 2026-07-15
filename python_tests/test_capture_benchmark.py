import json
from pathlib import Path

import numpy as np
import pytest

from delta_vision.benchmark import (
    CaptureBenchmark,
    CaptureMetrics,
    benchmark_source,
    evaluate_capture_metrics,
    run_windows_benchmark,
)
from delta_vision.benchmark import (
    main as benchmark_main,
)
from delta_vision.frames import CapturedFrame

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "controlled-window.json"


class FakeSource:
    def __init__(self, frames) -> None:
        self._frames = iter(frames)
        self.closed = False

    def grab(self):
        return next(self._frames)

    def close(self) -> None:
        self.closed = True


def _frame(sequence: int, shape: tuple[int, int, int]) -> CapturedFrame:
    image = np.zeros(shape, dtype=np.uint8)
    image.setflags(write=False)
    return CapturedFrame(sequence, sequence + 1, image, "fixture")


def test_capture_benchmark_reports_fps_p95_none_and_resolution_drift() -> None:
    source = FakeSource(
        [
            _frame(0, (100, 200, 3)),
            None,
            _frame(1, (120, 200, 3)),
        ]
    )
    clock = iter(
        [
            0,
            0,
            10_000_000,
            100_000_000,
            110_000_000,
            200_000_000,
            230_000_000,
            1_000_000_000,
            1_000_000_000,
        ]
    )

    benchmark = benchmark_source(
        source,
        duration_seconds=1,
        clock_ns=lambda: next(clock),
        foreground_probe=iter((True, False, True)).__next__,
    )

    assert benchmark.metrics.frame_count == 2
    assert benchmark.metrics.no_frame_count == 1
    assert benchmark.metrics.black_frame_count == 2
    assert benchmark.metrics.average_fps == 2
    assert benchmark.metrics.capture_latency_p95_ms == 30
    assert benchmark.metrics.initial_resolution == (200, 100)
    assert benchmark.metrics.resolution_drift_count == 1
    assert benchmark.metrics.foreground_mismatch_count == 1
    assert benchmark.first_frame.sequence == 0
    assert benchmark.last_frame.sequence == 1
    assert source.closed is True


def test_capture_benchmark_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="持续时间"):
        benchmark_source(FakeSource([]), duration_seconds=0)


def test_capture_benchmark_treats_one_bright_noise_pixel_as_near_black() -> None:
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[0, 0, 0] = 6
    image.setflags(write=False)
    source = FakeSource([CapturedFrame(0, 1, image, "fixture")])
    clock = iter([0, 0, 10_000_000, 1_000_000_000, 1_000_000_000])

    benchmark = benchmark_source(
        source,
        duration_seconds=1,
        clock_ns=lambda: next(clock),
    )

    assert benchmark.metrics.black_frame_count == 1


def test_capture_metrics_rejects_fps_inconsistent_with_frame_count() -> None:
    with pytest.raises(ValueError, match="average_fps"):
        CaptureMetrics(
            duration_seconds=60,
            frame_count=1,
            no_frame_count=0,
            black_frame_count=0,
            average_fps=20,
            capture_latency_average_ms=20,
            capture_latency_p95_ms=40,
            capture_latency_max_ms=45,
            initial_resolution=(1920, 1080),
            resolution_drift_count=0,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"frame_count": True}, "frame_count"),
        ({"capture_latency_average_ms": -1}, "capture_latency_average_ms"),
        ({"capture_latency_p95_ms": float("nan")}, "capture_latency_p95_ms"),
        ({"initial_resolution": ("invalid",)}, "initial_resolution"),
    ],
)
def test_capture_metrics_rejects_invalid_runtime_schema(override, message) -> None:
    values = {
        "duration_seconds": 60,
        "frame_count": 1200,
        "no_frame_count": 0,
        "black_frame_count": 0,
        "average_fps": 20,
        "capture_latency_average_ms": 20,
        "capture_latency_p95_ms": 40,
        "capture_latency_max_ms": 45,
        "initial_resolution": (1920, 1080),
        "resolution_drift_count": 0,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        CaptureMetrics(**values)


def test_capture_gate_rejects_missing_and_black_frames() -> None:
    metrics = CaptureMetrics(
        duration_seconds=60,
        frame_count=1200,
        no_frame_count=1,
        black_frame_count=1,
        average_fps=20,
        capture_latency_average_ms=20,
        capture_latency_p95_ms=40,
        capture_latency_max_ms=45,
        initial_resolution=(1920, 1080),
        resolution_drift_count=0,
    )

    result = evaluate_capture_metrics(metrics)

    assert result.passed is False
    assert result.failures == ("missing_frames", "black_frames")


def test_capture_gate_enforces_duration_fps_latency_and_resolution() -> None:
    metrics = CaptureMetrics(
        duration_seconds=58,
        frame_count=1102,
        no_frame_count=0,
        black_frame_count=0,
        average_fps=19,
        capture_latency_average_ms=20,
        capture_latency_p95_ms=51,
        capture_latency_max_ms=60,
        initial_resolution=(1920, 1080),
        resolution_drift_count=1,
        foreground_mismatch_count=1,
    )

    result = evaluate_capture_metrics(metrics)

    assert result.passed is False
    assert result.failures == (
        "duration_below_minimum",
        "fps_below_minimum",
        "latency_p95_above_maximum",
        "resolution_drift",
        "foreground_window_mismatch",
    )


def test_capture_gate_rejects_metrics_without_captured_resolution() -> None:
    metrics = CaptureMetrics(
        duration_seconds=60,
        frame_count=0,
        no_frame_count=0,
        black_frame_count=0,
        average_fps=0,
        capture_latency_average_ms=20,
        capture_latency_p95_ms=40,
        capture_latency_max_ms=45,
        initial_resolution=None,
        resolution_drift_count=0,
    )

    result = evaluate_capture_metrics(metrics)

    assert result.passed is False
    assert result.failures == (
        "no_captured_frames",
        "missing_initial_resolution",
        "fps_below_minimum",
    )


def test_benchmark_cli_persists_machine_readable_gate_failure(
    tmp_path, monkeypatch
) -> None:
    metrics = CaptureMetrics(
        duration_seconds=60,
        frame_count=1200,
        no_frame_count=0,
        black_frame_count=1,
        average_fps=20,
        capture_latency_average_ms=20,
        capture_latency_p95_ms=40,
        capture_latency_max_ms=45,
        initial_resolution=(1920, 1080),
        resolution_drift_count=0,
    )
    monkeypatch.setattr(
        "delta_vision.benchmark.run_windows_benchmark",
        lambda **_kwargs: CaptureBenchmark(metrics, None, None),
    )

    exit_code = benchmark_main(
        [
            "--window-title",
            "三角洲行动",
            "--duration",
            "60",
            "--artifacts",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    gate = json.loads((tmp_path / "capture-gate.json").read_text(encoding="utf-8"))
    assert gate == {
        "schema_version": 1,
        "run_id": None,
        "passed": False,
        "failures": ["black_frames"],
    }


def test_windows_benchmark_binds_run_and_persisted_frame_hashes(
    tmp_path, monkeypatch
) -> None:
    frame = _frame(0, (100, 200, 3))
    metrics = CaptureMetrics(
        duration_seconds=60,
        frame_count=1200,
        no_frame_count=0,
        black_frame_count=0,
        average_fps=20,
        capture_latency_average_ms=20,
        capture_latency_p95_ms=40,
        capture_latency_max_ms=45,
        initial_resolution=(200, 100),
        resolution_drift_count=0,
        run_id="benchmark-run",
    )
    monkeypatch.setattr(
        "delta_vision.benchmark.window_client_region",
        lambda _title: object(),
    )
    monkeypatch.setattr(
        "delta_vision.benchmark.find_window_handle",
        lambda _title: 123,
    )
    monkeypatch.setattr(
        "delta_vision.benchmark.DxcamFrameSource",
        lambda _region: FakeSource([]),
    )
    monkeypatch.setattr(
        "delta_vision.benchmark.Win32NativeGateway",
        lambda: type("Gateway", (), {"foreground_window_handle": lambda self: 123})(),
    )
    monkeypatch.setattr(
        "delta_vision.benchmark.benchmark_source",
        lambda *_args, **_kwargs: CaptureBenchmark(metrics, frame, frame),
    )

    result = run_windows_benchmark(
        window_title="三角洲行动",
        backend="dxcam",
        duration_seconds=60,
        artifacts=tmp_path,
        run_id="benchmark-run",
    )

    persisted = json.loads(
        (tmp_path / "capture-metrics.json").read_text(encoding="utf-8")
    )
    assert persisted["run_id"] == "benchmark-run"
    assert persisted["first_frame_sha256"] == result.metrics.first_frame_sha256
    assert persisted["last_frame_sha256"] == result.metrics.last_frame_sha256
    assert persisted["first_frame_sha256"]
    assert persisted["last_frame_sha256"]


def test_benchmark_cli_resolves_window_and_backend_from_worker_config(
    tmp_path, monkeypatch
) -> None:
    metrics = CaptureMetrics(
        duration_seconds=60,
        frame_count=1200,
        no_frame_count=0,
        black_frame_count=0,
        average_fps=20,
        capture_latency_average_ms=20,
        capture_latency_p95_ms=40,
        capture_latency_max_ms=45,
        initial_resolution=(800, 600),
        resolution_drift_count=0,
    )
    arguments = {}

    def run_benchmark(**kwargs):
        arguments.update(kwargs)
        return CaptureBenchmark(metrics, None, None)

    monkeypatch.setattr(
        "delta_vision.benchmark.run_windows_benchmark",
        run_benchmark,
    )

    exit_code = benchmark_main(
        [
            "--config",
            str(CONFIG_PATH),
            "--duration",
            "60",
            "--artifacts",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert arguments["window_title"] == "Delta Vision Test Target"
    assert arguments["backend"] == "dxcam"
