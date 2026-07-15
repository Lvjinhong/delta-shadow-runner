import numpy as np
import pytest

from delta_vision.benchmark import benchmark_source
from delta_vision.frames import CapturedFrame


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
    )

    assert benchmark.metrics.frame_count == 2
    assert benchmark.metrics.no_frame_count == 1
    assert benchmark.metrics.average_fps == 2
    assert benchmark.metrics.capture_latency_p95_ms == 30
    assert benchmark.metrics.initial_resolution == (200, 100)
    assert benchmark.metrics.resolution_drift_count == 1
    assert benchmark.first_frame.sequence == 0
    assert benchmark.last_frame.sequence == 1
    assert source.closed is True


def test_capture_benchmark_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="持续时间"):
        benchmark_source(FakeSource([]), duration_seconds=0)
