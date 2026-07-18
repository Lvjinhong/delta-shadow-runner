import json
from collections.abc import Iterable

import numpy as np
import pytest

from delta_vision.actuator import DryRunActuator
from delta_vision.config import CaptureRegion
from delta_vision.events import JsonlEventWriter
from delta_vision.frames import CapturedFrame, FrameRecorder
from delta_vision.menu_automation import (
    MenuActionKind,
    MenuControllerStatus,
    MenuScene,
    MenuTransition,
    SceneDecisionReason,
    SceneObservation,
    VisualMenuController,
)
from delta_vision.menu_runtime import MenuActionExecutor
from delta_vision.menu_worker import run_menu_control_loop


class _FrameSource:
    def __init__(self, scenes: Iterable[MenuScene | None]) -> None:
        self._scenes = iter(scenes)
        self.closed = False
        self.sequence = 0

    def grab(self) -> CapturedFrame | None:
        try:
            scene = next(self._scenes)
        except StopIteration:
            return None
        if scene is None:
            return None
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        image.setflags(write=False)
        frame = CapturedFrame(
            sequence=self.sequence,
            captured_at_ns=1_000 + self.sequence,
            image=image,
            source="fixture",
            metadata={"scene": str(scene)},
        )
        self.sequence += 1
        return frame

    def close(self) -> None:
        self.closed = True


class _Observer:
    def observe(self, frame: CapturedFrame) -> SceneObservation:
        scene = MenuScene(frame.metadata["scene"])
        action_point = (20.0, 30.0) if scene is MenuScene.LOBBY else None
        return SceneObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            scene=scene,
            candidate_scene=scene,
            confidence=0.99,
            runner_up_confidence=0.1,
            accepted=True,
            reason=SceneDecisionReason.ACCEPTED,
            action_accepted=action_point is not None,
            action_point=action_point,
            page_point=(10.0, 10.0),
            page_template_id=f"{scene}-page",
        )


class _FailingObserver:
    def observe(self, _frame: CapturedFrame) -> SceneObservation:
        raise RuntimeError("vision failed")


class _Clock:
    def __init__(self, values: Iterable[int]) -> None:
        self._values = iter(values)
        self._last = 0

    def __call__(self) -> int:
        try:
            self._last = next(self._values)
        except StopIteration:
            self._last += 1
        return self._last


class _FailingClock:
    def __call__(self) -> int:
        raise RuntimeError("clock failed")


class _PressedKeysFailingActuator:
    def __init__(self, delegate: DryRunActuator) -> None:
        self._delegate = delegate

    @property
    def pressed_keys(self):
        raise RuntimeError("secondary pressed keys failure")

    @property
    def events(self):
        return self._delegate.events

    def expire_overdue(self, *, now_ns):
        return self._delegate.expire_overdue(now_ns=now_ns)

    def release_all(self, *, now_ns, reason):
        return self._delegate.release_all(now_ns=now_ns, reason=reason)


def _controller() -> VisualMenuController:
    return VisualMenuController(
        transitions=(
            MenuTransition(
                source=MenuScene.LOBBY,
                target=MenuScene.IN_MATCH,
                action_kind=MenuActionKind.CLICK,
            ),
        ),
        confirmation_frames=3,
        maximum_confirmation_span_ms=750,
        maximum_action_point_drift_px=12,
        maximum_page_point_drift_px=12,
        maximum_frame_age_ms=1_000,
        transition_timeout_ms=8_000,
        stop_scenes=frozenset({MenuScene.DEATH_SUMMARY}),
    )


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_menu_loop_records_confirmed_action_and_completes(tmp_path) -> None:
    source = _FrameSource(
        [MenuScene.LOBBY] * 3 + [MenuScene.IN_MATCH] * 3
    )
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)
    recorder = FrameRecorder(tmp_path / "replay")
    event_writer = JsonlEventWriter(
        tmp_path / "events.jsonl",
        run_id="menu-run",
        truncate=True,
    )

    result = run_menu_control_loop(
        source=source,
        observer=_Observer(),
        controller=_controller(),
        executor=MenuActionExecutor(
            actuator=actuator,
            capture_region=CaptureRegion(100, 200, 80, 60),
            clock_ns=lambda: 1_100,
        ),
        actuator=actuator,
        recorder=recorder,
        event_writer=event_writer,
        loop_interval_ms=1,
        max_duration_seconds=10,
        clock_ns=_Clock(range(1_100, 1_500)),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is MenuControllerStatus.COMPLETED
    assert result.frame_count == 6
    assert result.action_count == 1
    assert source.closed is True
    assert actuator.pressed_keys == frozenset()
    inputs = _records(tmp_path / "replay" / "input-events.jsonl")
    assert [record["payload"]["kind"] for record in inputs] == [
        "mouse_move_absolute",
        "mouse_left_down",
        "mouse_left_up",
    ]
    assert inputs[0]["payload"]["x"] == 120
    assert inputs[0]["payload"]["y"] == 230
    events = _records(tmp_path / "events.jsonl")
    assert all(record["run_id"] == "menu-run" for record in events)
    assert events[-1]["event_type"] == "menu_terminal"
    assert events[-1]["payload"]["status"] == "completed"


def test_menu_loop_timeout_releases_input_and_closes_source(tmp_path) -> None:
    source = _FrameSource([None, None])
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)

    result = run_menu_control_loop(
        source=source,
        observer=_Observer(),
        controller=_controller(),
        executor=MenuActionExecutor(
            actuator=actuator,
            capture_region=CaptureRegion(0, 0, 80, 60),
            clock_ns=lambda: 2_000_000_000,
        ),
        actuator=actuator,
        recorder=FrameRecorder(tmp_path / "replay"),
        event_writer=JsonlEventWriter(tmp_path / "events.jsonl", truncate=True),
        loop_interval_ms=1,
        max_duration_seconds=1,
        clock_ns=_Clock([0, 100, 2_000_000_000, 2_000_000_001]),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is MenuControllerStatus.STOPPED
    assert result.reason == "菜单 Worker 运行超时"
    assert result.frame_count == 0
    assert result.action_count == 0
    assert source.closed is True
    assert actuator.pressed_keys == frozenset()


def test_menu_loop_exception_releases_input_closes_source_and_records_error(tmp_path) -> None:
    source = _FrameSource([MenuScene.LOBBY])
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)
    events_path = tmp_path / "events.jsonl"

    with pytest.raises(RuntimeError, match="vision failed"):
        run_menu_control_loop(
            source=source,
            observer=_FailingObserver(),
            controller=_controller(),
            executor=MenuActionExecutor(
                actuator=actuator,
                capture_region=CaptureRegion(0, 0, 80, 60),
                clock_ns=lambda: 1_100,
            ),
            actuator=actuator,
            recorder=FrameRecorder(tmp_path / "replay"),
            event_writer=JsonlEventWriter(events_path, truncate=True),
            loop_interval_ms=1,
            max_duration_seconds=10,
            clock_ns=_Clock(range(1_100, 1_500)),
            sleep_fn=lambda _seconds: None,
        )

    assert source.closed is True
    assert actuator.pressed_keys == frozenset()
    assert _records(events_path)[-1]["event_type"] == "runtime_error"


def test_menu_loop_validation_failure_still_releases_and_closes(tmp_path) -> None:
    source = _FrameSource([MenuScene.LOBBY])
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)
    events_path = tmp_path / "events.jsonl"

    with pytest.raises(ValueError, match="循环间隔"):
        run_menu_control_loop(
            source=source,
            observer=_Observer(),
            controller=_controller(),
            executor=MenuActionExecutor(
                actuator=actuator,
                capture_region=CaptureRegion(0, 0, 80, 60),
            ),
            actuator=actuator,
            recorder=FrameRecorder(tmp_path / "replay"),
            event_writer=JsonlEventWriter(events_path, truncate=True),
            loop_interval_ms=0,
            max_duration_seconds=10,
        )

    assert source.closed is True
    assert actuator.pressed_keys == frozenset()
    assert _records(events_path)[-1]["event_type"] == "runtime_error"


def test_menu_loop_first_clock_failure_uses_safe_cleanup_time(tmp_path) -> None:
    source = _FrameSource([MenuScene.LOBBY])
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)

    with pytest.raises(RuntimeError, match="clock failed"):
        run_menu_control_loop(
            source=source,
            observer=_Observer(),
            controller=_controller(),
            executor=MenuActionExecutor(
                actuator=actuator,
                capture_region=CaptureRegion(0, 0, 80, 60),
            ),
            actuator=actuator,
            recorder=FrameRecorder(tmp_path / "replay"),
            event_writer=JsonlEventWriter(tmp_path / "events.jsonl", truncate=True),
            loop_interval_ms=1,
            max_duration_seconds=10,
            clock_ns=_FailingClock(),
        )

    assert source.closed is True
    assert actuator.pressed_keys == frozenset()


def test_menu_loop_cleanup_property_failure_does_not_mask_original_error(tmp_path) -> None:
    source = _FrameSource([MenuScene.LOBBY])
    delegate = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)
    actuator = _PressedKeysFailingActuator(delegate)
    events_path = tmp_path / "events.jsonl"

    with pytest.raises(RuntimeError, match="vision failed"):
        run_menu_control_loop(
            source=source,
            observer=_FailingObserver(),
            controller=_controller(),
            executor=MenuActionExecutor(
                actuator=delegate,
                capture_region=CaptureRegion(0, 0, 80, 60),
                clock_ns=lambda: 1_100,
            ),
            actuator=actuator,
            recorder=FrameRecorder(tmp_path / "replay"),
            event_writer=JsonlEventWriter(events_path, truncate=True),
            loop_interval_ms=1,
            max_duration_seconds=10,
            clock_ns=_Clock(range(1_100, 1_500)),
            sleep_fn=lambda _seconds: None,
        )

    assert source.closed is True
    error_event = _records(events_path)[-1]
    assert error_event["event_type"] == "runtime_error"
    assert error_event["payload"]["error"] == "vision failed"
    assert error_event["payload"]["pressed_keys"] is None
    assert any(
        "actuator.pressed_keys" in message
        for message in error_event["payload"]["cleanup_errors"]
    )
