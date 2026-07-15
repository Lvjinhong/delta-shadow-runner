import json

import numpy as np
import pytest

from delta_vision import sample_frames
from delta_vision.config import CaptureRegion
from delta_vision.frames import CapturedFrame, ReplayFrameSource
from delta_vision.sample_frames import (
    FrameSamplingSchedule,
    SamplingResult,
    run_windows_sampling,
    sample_source,
)


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += round(seconds * 1_000_000_000)


class FakeSource:
    def __init__(self, frames) -> None:
        self._frames = iter(frames)
        self.closed = False

    def grab(self):
        return next(self._frames)

    def close(self) -> None:
        self.closed = True


def _frame(sequence: int, *, width: int = 80, height: int = 60) -> CapturedFrame:
    image = np.full((height, width, 3), sequence, dtype=np.uint8)
    image.setflags(write=False)
    return CapturedFrame(sequence, 1_000 + sequence, image, "fixture")


def test_sampling_schedule_uses_exact_five_hz_boundaries() -> None:
    schedule = FrameSamplingSchedule(started_at_ns=0, duration_seconds=1, sample_fps=5)

    assert schedule.consume(0) is True
    assert schedule.consume(199_999_999) is False
    assert schedule.consume(200_000_000) is True
    assert schedule.consume(999_999_999) is True
    assert schedule.consume(1_000_000_000) is False


@pytest.mark.parametrize(
    ("started_at_ns", "duration_seconds", "sample_fps"),
    [
        (-1, 1, 2),
        (True, 1, 2),
        (0, 0, 2),
        (0, float("inf"), 2),
        (0, 1, 1),
        (0, 1, 6),
        (0, True, 2),
        (0, 1, True),
    ],
)
def test_sampling_schedule_rejects_invalid_policy(
    started_at_ns: object, duration_seconds: object, sample_fps: object
) -> None:
    with pytest.raises(ValueError):
        FrameSamplingSchedule(
            started_at_ns=started_at_ns,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
        )


def test_sampling_schedule_rejects_non_monotonic_clock() -> None:
    schedule = FrameSamplingSchedule(started_at_ns=10, duration_seconds=1, sample_fps=2)

    with pytest.raises(ValueError, match="单调递增"):
        schedule.consume(9)


def test_sample_source_records_traceable_fixed_resolution_run(tmp_path) -> None:
    clock = FakeClock()
    source = FakeSource([_frame(0), _frame(1), _frame(2)])

    result = sample_source(
        source,
        output_directory=tmp_path / "run",
        run_id="game-route-001",
        dataset_split="calibration",
        window_title="三角洲行动",
        backend="dxcam",
        duration_seconds=1.1,
        sample_fps=2,
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )

    assert source.closed is True
    assert result.frame_count == 3
    assert result.no_frame_count == 0
    assert result.resolution == (80, 60)
    replay = list(ReplayFrameSource(tmp_path / "run"))
    assert [frame.sequence for frame in replay] == [0, 1, 2]
    assert replay[0].metadata["run_id"] == "game-route-001"
    assert replay[0].metadata["dataset_kind"] == "manual-game-route"
    assert replay[0].metadata["dataset_split"] == "calibration"
    report = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert report["window_title"] == "三角洲行动"
    assert report["backend"] == "dxcam"
    assert report["sample_fps"] == 2
    assert report["resolution"] == [80, 60]
    assert report["dataset_split"] == "calibration"


@pytest.mark.parametrize("dataset_split", ["", "test", "CALIBRATION"])
def test_sample_source_rejects_invalid_dataset_split(tmp_path, dataset_split: str) -> None:
    source = FakeSource([])

    with pytest.raises(ValueError, match="dataset_split"):
        sample_source(
            source,
            output_directory=tmp_path / "run",
            run_id="game-route-split",
            dataset_split=dataset_split,
            window_title="三角洲行动",
            backend="dxcam",
            duration_seconds=1,
            sample_fps=2,
        )


def test_sample_source_counts_none_frames_and_closes_source(tmp_path) -> None:
    clock = FakeClock()
    source = FakeSource([None, _frame(1), None])

    result = sample_source(
        source,
        output_directory=tmp_path / "run",
        run_id="game-route-002",
        dataset_split="calibration",
        window_title="三角洲行动",
        backend="mss",
        duration_seconds=1.1,
        sample_fps=2,
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )

    assert result.frame_count == 1
    assert result.no_frame_count == 2
    assert source.closed is True


def test_sample_source_rejects_run_without_usable_frame(tmp_path) -> None:
    clock = FakeClock()
    source = FakeSource([None, None])

    with pytest.raises(RuntimeError, match="没有获得任何可用截图"):
        sample_source(
            source,
            output_directory=tmp_path / "run",
            run_id="game-route-empty",
            dataset_split="calibration",
            window_title="三角洲行动",
            backend="dxcam",
            duration_seconds=0.6,
            sample_fps=2,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert source.closed is True


def test_sample_source_rejects_resolution_drift_and_closes_source(tmp_path) -> None:
    clock = FakeClock()
    source = FakeSource([_frame(0), _frame(1, width=81)])

    with pytest.raises(ValueError, match="分辨率漂移"):
        sample_source(
            source,
            output_directory=tmp_path / "run",
            run_id="game-route-003",
            dataset_split="calibration",
            window_title="三角洲行动",
            backend="dxcam",
            duration_seconds=0.6,
            sample_fps=2,
            clock_ns=clock,
            sleep_fn=clock.sleep,
        )

    assert source.closed is True


@pytest.mark.parametrize("run_id", ["", "has space", "../escape", "a" * 129])
def test_sample_source_rejects_unsafe_run_id(tmp_path, run_id: str) -> None:
    source = FakeSource([])

    with pytest.raises(ValueError, match="run_id"):
        sample_source(
            source,
            output_directory=tmp_path / "run",
            run_id=run_id,
            dataset_split="calibration",
            window_title="三角洲行动",
            backend="dxcam",
            duration_seconds=1,
            sample_fps=2,
        )


def test_sample_source_refuses_to_append_to_existing_run(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.jsonl").write_text("existing\n", encoding="utf-8")
    source = FakeSource([])

    with pytest.raises(FileExistsError, match="已经存在"):
        sample_source(
            source,
            output_directory=output,
            run_id="game-route-004",
            dataset_split="calibration",
            window_title="三角洲行动",
            backend="dxcam",
            duration_seconds=1,
            sample_fps=2,
        )


@pytest.mark.parametrize(
    ("window_title", "backend"),
    [("", "dxcam"), ("三角洲行动", "other")],
)
def test_sample_source_rejects_invalid_target(tmp_path, window_title: str, backend: str) -> None:
    source = FakeSource([])

    with pytest.raises(ValueError):
        sample_source(
            source,
            output_directory=tmp_path / "run",
            run_id="game-route-invalid",
            dataset_split="calibration",
            window_title=window_title,
            backend=backend,
            duration_seconds=1,
            sample_fps=2,
        )

    assert source.closed is True


@pytest.mark.parametrize("start_delay", [-1, 301, float("inf"), True])
def test_run_windows_sampling_rejects_invalid_start_delay(tmp_path, start_delay: object) -> None:
    with pytest.raises(ValueError, match="开始倒计时"):
        run_windows_sampling(
            window_title="三角洲行动",
            backend="dxcam",
            output_directory=tmp_path,
            run_id="game-route-delay",
            dataset_split="calibration",
            duration_seconds=1,
            sample_fps=2,
            start_delay_seconds=start_delay,
        )


@pytest.mark.parametrize("backend", ["dxcam", "mss"])
def test_run_windows_sampling_selects_backend_and_forwards_contract(
    tmp_path, monkeypatch, backend: str
) -> None:
    source = FakeSource([])
    calls = []
    expected = SamplingResult(
        run_id="game-route-windows",
        dataset_split="validation",
        window_title="三角洲行动",
        backend=backend,
        requested_duration_seconds=1,
        measured_duration_seconds=1,
        sample_fps=2,
        frame_count=1,
        no_frame_count=0,
        resolution=(80, 60),
    )

    def fake_sample(actual_source, **kwargs):
        calls.append((actual_source, kwargs))
        return expected

    monkeypatch.setattr(sample_frames, "sample_source", fake_sample)
    sleeps = []
    result = run_windows_sampling(
        window_title="三角洲行动",
        backend=backend,
        output_directory=tmp_path,
        run_id="game-route-windows",
        dataset_split="validation",
        duration_seconds=1,
        sample_fps=2,
        start_delay_seconds=3,
        sleep_fn=sleeps.append,
        region_resolver=lambda title: CaptureRegion(1, 2, 80, 60),
        dxcam_factory=lambda region: source,
        mss_factory=lambda region: source,
    )

    assert result is expected
    assert sleeps == [3]
    assert calls[0][0] is source
    assert calls[0][1]["backend"] == backend
    assert calls[0][1]["dataset_split"] == "validation"


def test_run_windows_sampling_rejects_unknown_backend_after_region_resolution(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="backend"):
        run_windows_sampling(
            window_title="三角洲行动",
            backend="other",
            output_directory=tmp_path,
            run_id="game-route-backend",
            dataset_split="calibration",
            duration_seconds=1,
            sample_fps=2,
            start_delay_seconds=0,
            region_resolver=lambda title: CaptureRegion(0, 0, 80, 60),
        )


def test_sample_frames_main_reports_success(monkeypatch, capsys, tmp_path) -> None:
    result = SamplingResult(
        run_id="game-route-cli",
        dataset_split="blind",
        window_title="三角洲行动",
        backend="dxcam",
        requested_duration_seconds=1,
        measured_duration_seconds=1,
        sample_fps=2,
        frame_count=1,
        no_frame_count=0,
        resolution=(80, 60),
    )
    monkeypatch.setattr(sample_frames, "run_windows_sampling", lambda **kwargs: result)

    exit_code = sample_frames.main(
        [
            "--window-title",
            "三角洲行动",
            "--output",
            str(tmp_path),
            "--split",
            "blind",
            "--duration",
            "1",
            "--fps",
            "2",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["frame_count"] == 1


def test_sample_frames_main_reports_runtime_error(monkeypatch, capsys, tmp_path) -> None:
    def fail(**kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(sample_frames, "run_windows_sampling", fail)

    exit_code = sample_frames.main(
        [
            "--window-title",
            "三角洲行动",
            "--output",
            str(tmp_path),
            "--split",
            "blind",
        ]
    )

    assert exit_code == 1
    assert "capture failed" in capsys.readouterr().err
