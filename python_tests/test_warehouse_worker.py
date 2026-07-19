import json
from pathlib import Path

import numpy as np
import pytest

from delta_vision.config import CaptureRegion
from delta_vision.events import JsonlEventWriter
from delta_vision.frames import CapturedFrame, FrameRecorder
from delta_vision.warehouse_cleanup import (
    SafeSlotState,
    WarehouseCleanupController,
    WarehouseCleanupPolicy,
    WarehouseCleanupStatus,
    WarehouseObservation,
    WarehouseScene,
)
from delta_vision.warehouse_worker import (
    DryRunCleanupExecutor,
    DuplicateCleanupActionError,
    ExpiredCleanupActionError,
    run_warehouse_cleanup_loop,
)


class MutableClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns


class SequenceSource:
    def __init__(self, scenes: list[WarehouseScene], clock: MutableClock) -> None:
        self.frames = [
            CapturedFrame(
                sequence=index,
                captured_at_ns=100_000_000 + index * 100_000_000,
                image=np.full((1080, 1920, 3), index, dtype=np.uint8),
                source="warehouse-replay",
                metadata={"scene": str(scene)},
            )
            for index, scene in enumerate(scenes)
        ]
        self.clock = clock
        self.closed = False

    def grab(self) -> CapturedFrame | None:
        if not self.frames:
            return None
        frame = self.frames.pop(0)
        self.clock.now_ns = frame.captured_at_ns + 20_000_000
        return frame

    def close(self) -> None:
        self.closed = True


class MetadataObserver:
    def observe(self, frame: CapturedFrame) -> WarehouseObservation:
        scene = WarehouseScene(frame.metadata["scene"])
        if scene is WarehouseScene.BASE:
            return WarehouseObservation(
                frame_sequence=frame.sequence,
                captured_at_ns=frame.captured_at_ns,
                frame_size=(1920, 1080),
                scene=scene,
                accepted=True,
                open_warehouse_point=(322.5, 55.0),
            )
        if scene is WarehouseScene.WAREHOUSE:
            return WarehouseObservation(
                frame_sequence=frame.sequence,
                captured_at_ns=frame.captured_at_ns,
                frame_size=(1920, 1080),
                scene=scene,
                accepted=True,
                safe_box_count=0,
                slots=(SafeSlotState.EMPTY, SafeSlotState.EMPTY),
                return_base_point=(192.5, 55.0),
            )
        return WarehouseObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            frame_size=(1920, 1080),
            scene=WarehouseScene.UNKNOWN,
            accepted=False,
        )


def _run(tmp_path: Path, scenes: list[WarehouseScene]):
    clock = MutableClock()
    source = SequenceSource(scenes, clock)
    executor = DryRunCleanupExecutor(
        capture_region=CaptureRegion(left=320, top=156, width=1920, height=1080),
        clock_ns=clock,
    )
    result = run_warehouse_cleanup_loop(
        source=source,
        observer=MetadataObserver(),
        controller=WarehouseCleanupController(WarehouseCleanupPolicy()),
        executor=executor,
        recorder=FrameRecorder(tmp_path / "replay"),
        event_writer=JsonlEventWriter(
            tmp_path / "events.jsonl",
            run_id="warehouse-dry-run",
            truncate=True,
        ),
        loop_interval_ms=20,
        max_duration_seconds=10,
        clock_ns=clock,
        sleep_fn=lambda _seconds: None,
    )
    return result, source, executor


def test_realistic_empty_cleanup_replay_completes_without_os_input(tmp_path: Path) -> None:
    result, source, executor = _run(
        tmp_path,
        [WarehouseScene.BASE] * 3
        + [WarehouseScene.WAREHOUSE] * 3
        + [WarehouseScene.BASE] * 3,
    )

    assert result.status is WarehouseCleanupStatus.COMPLETED
    assert result.frame_count == 9
    assert result.action_count == 2
    assert result.safe_box_count == 0
    assert source.closed is True
    assert [record.intent_id for record in executor.records] == [
        "open-warehouse",
        "return-base",
    ]
    assert [record.local_position for record in executor.records] == [
        (322, 55),
        (192, 55),
    ]
    assert [record.screen_position for record in executor.records] == [
        (642, 211),
        (512, 211),
    ]
    assert not (tmp_path / "replay/input-events.jsonl").exists()

    frame_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "warehouse_frame"
    ]
    assert len(frame_events) == 9
    assert frame_events[-1]["payload"]["status"] == "completed"


def test_unknown_frame_stops_without_dry_run_action(tmp_path: Path) -> None:
    result, source, executor = _run(tmp_path, [WarehouseScene.UNKNOWN])

    assert result.status is WarehouseCleanupStatus.STOPPED
    assert result.action_count == 0
    assert "未知" in (result.reason or "")
    assert executor.records == ()
    assert source.closed is True


def test_dry_run_executor_rejects_expired_and_duplicate_intents() -> None:
    clock = MutableClock()
    executor = DryRunCleanupExecutor(
        capture_region=CaptureRegion(0, 0, 1920, 1080),
        clock_ns=clock,
    )
    controller = WarehouseCleanupController(
        WarehouseCleanupPolicy(confirmation_frames=1)
    )
    observation = WarehouseObservation(
        frame_sequence=0,
        captured_at_ns=100,
        frame_size=(1920, 1080),
        scene=WarehouseScene.BASE,
        accepted=True,
        open_warehouse_point=(322.5, 55.0),
    )
    snapshot = controller.step(observation, now_ns=100)
    assert snapshot.action is not None

    with pytest.raises(ExpiredCleanupActionError):
        executor.execute(snapshot.action, now_ns=snapshot.action.expires_at_ns)

    executor.execute(snapshot.action, now_ns=101)
    with pytest.raises(DuplicateCleanupActionError):
        executor.execute(snapshot.action, now_ns=102)


def test_source_close_failure_is_not_retried(tmp_path: Path) -> None:
    clock = MutableClock()

    class FailingCloseSource(SequenceSource):
        def __init__(self) -> None:
            super().__init__([WarehouseScene.UNKNOWN], clock)
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("close-failure")

    source = FailingCloseSource()

    with pytest.raises(RuntimeError, match="close-failure"):
        run_warehouse_cleanup_loop(
            source=source,
            observer=MetadataObserver(),
            controller=WarehouseCleanupController(WarehouseCleanupPolicy()),
            executor=DryRunCleanupExecutor(
                capture_region=CaptureRegion(0, 0, 1920, 1080),
                clock_ns=clock,
            ),
            recorder=FrameRecorder(tmp_path / "replay"),
            event_writer=JsonlEventWriter(
                tmp_path / "events.jsonl",
                run_id="close-failure",
                truncate=True,
            ),
            loop_interval_ms=20,
            max_duration_seconds=10,
            clock_ns=clock,
            sleep_fn=lambda _seconds: None,
        )

    assert source.close_calls == 1
