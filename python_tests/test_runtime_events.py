import json
from dataclasses import dataclass

import pytest

import delta_vision.events as events_module
import delta_vision.frames as frames_module
from delta_vision.events import JsonlEventWriter
from delta_vision.frames import FrameRecorder
from delta_vision.runtime_events import persist_new_input_events


@dataclass(frozen=True)
class _InputEvent:
    kind: str
    at_ns: int
    key: str | None = None
    dx: int | None = None
    dy: int | None = None
    x: int | None = None
    y: int | None = None
    reason: str | None = None


class _Actuator:
    def __init__(self, events: tuple[_InputEvent, ...]) -> None:
        self.events = events


class _FailOnceAfterWrite:
    def __init__(self, writer: JsonlEventWriter) -> None:
        self._writer = writer
        self._failed = False

    def write(self, event, *, idempotency_key=None) -> None:
        self._writer.write(event, idempotency_key=idempotency_key)
        if not self._failed:
            self._failed = True
            raise OSError("runtime writer failed after append")


def test_persist_input_events_keeps_absolute_coordinates_and_cursor(tmp_path) -> None:
    actuator = _Actuator(
        (
            _InputEvent("mouse_move_absolute", 100, x=-80, y=720),
            _InputEvent("mouse_left_down", 101),
            _InputEvent("mouse_left_up", 102, reason="点击完成"),
        )
    )
    recorder = FrameRecorder(tmp_path / "replay")
    writer = JsonlEventWriter(tmp_path / "events.jsonl", run_id="menu-run", truncate=True)

    cursor, payloads = persist_new_input_events(
        actuator=actuator,
        input_event_cursor=0,
        recorder=recorder,
        event_writer=writer,
    )
    unchanged_cursor, unchanged_payloads = persist_new_input_events(
        actuator=actuator,
        input_event_cursor=cursor,
        recorder=recorder,
        event_writer=writer,
    )

    assert cursor == 3
    assert payloads[0]["x"] == -80
    assert payloads[0]["y"] == 720
    assert payloads[0]["dx"] is None
    assert [payload["kind"] for payload in payloads] == [
        "mouse_move_absolute",
        "mouse_left_down",
        "mouse_left_up",
    ]
    assert unchanged_cursor == cursor
    assert unchanged_payloads == []


def test_partial_double_write_can_resume_without_duplicate_input_events(tmp_path) -> None:
    actuator = _Actuator(
        (
            _InputEvent("mouse_move_absolute", 100, x=20, y=30),
            _InputEvent("mouse_left_down", 101),
        )
    )
    recorder = FrameRecorder(tmp_path / "replay")
    writer = _FailOnceAfterWrite(
        JsonlEventWriter(tmp_path / "events.jsonl", truncate=True)
    )

    with pytest.raises(OSError, match="after append"):
        persist_new_input_events(
            actuator=actuator,
            input_event_cursor=0,
            recorder=recorder,
            event_writer=writer,
        )

    cursor, payloads = persist_new_input_events(
        actuator=actuator,
        input_event_cursor=0,
        recorder=recorder,
        event_writer=writer,
    )

    replay_records = [
        json.loads(line)
        for line in (tmp_path / "replay" / "input-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    runtime_records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert cursor == 2
    assert len(payloads) == 2
    assert [record["payload"]["event_id"] for record in replay_records] == [
        "input:0:100:mouse_move_absolute",
        "input:1:101:mouse_left_down",
    ]
    assert [record["payload"]["event_id"] for record in runtime_records] == [
        "input:0:100:mouse_move_absolute",
        "input:1:101:mouse_left_down",
    ]


def test_reopened_sinks_restore_input_ids_sequence_and_time_watermark(tmp_path) -> None:
    first_actuator = _Actuator((_InputEvent("mouse_move_absolute", 100, x=20, y=30),))
    replay_path = tmp_path / "replay"
    events_path = tmp_path / "events.jsonl"
    persist_new_input_events(
        actuator=first_actuator,
        input_event_cursor=0,
        recorder=FrameRecorder(replay_path),
        event_writer=JsonlEventWriter(events_path, truncate=True),
    )
    second_actuator = _Actuator(
        (
            _InputEvent("mouse_move_absolute", 100, x=20, y=30),
            _InputEvent("mouse_left_down", 101),
        )
    )

    cursor, _ = persist_new_input_events(
        actuator=second_actuator,
        input_event_cursor=0,
        recorder=FrameRecorder(replay_path),
        event_writer=JsonlEventWriter(events_path),
    )

    replay_records = [
        json.loads(line)
        for line in (replay_path / "input-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    runtime_records = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert cursor == 2
    assert [record["sequence"] for record in replay_records] == [0, 1]
    assert len({record["event_id"] for record in replay_records}) == 2
    assert len({record["idempotency_key"] for record in runtime_records}) == 2


@pytest.mark.parametrize("failing_sink", ["recorder", "runtime"])
def test_mid_append_failure_repairs_tail_before_old_cursor_retry(
    tmp_path, monkeypatch, failing_sink
) -> None:
    actuator = _Actuator(
        (
            _InputEvent("mouse_move_absolute", 100, x=20, y=30),
            _InputEvent("mouse_left_down", 101),
        )
    )
    recorder = FrameRecorder(tmp_path / "replay")
    writer = JsonlEventWriter(tmp_path / "events.jsonl", truncate=True)
    module = frames_module if failing_sink == "recorder" else events_module
    original_append = module.append_jsonl_record

    def append_half_then_fail(path, serialized) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized[: len(serialized) // 2])
        raise OSError(f"{failing_sink} mid append")

    monkeypatch.setattr(module, "append_jsonl_record", append_half_then_fail)
    with pytest.raises(OSError, match="mid append"):
        persist_new_input_events(
            actuator=actuator,
            input_event_cursor=0,
            recorder=recorder,
            event_writer=writer,
        )
    monkeypatch.setattr(module, "append_jsonl_record", original_append)

    cursor, _ = persist_new_input_events(
        actuator=actuator,
        input_event_cursor=0,
        recorder=recorder,
        event_writer=writer,
    )

    replay_records = [
        json.loads(line)
        for line in (tmp_path / "replay" / "input-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    runtime_records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert cursor == 2
    assert [record["sequence"] for record in replay_records] == [0, 1]
    assert len(runtime_records) == 2
