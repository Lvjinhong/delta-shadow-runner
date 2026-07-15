import json
import math

import pytest

from delta_vision.events import JsonlEventWriter, RuntimeEvent


def test_jsonl_event_writer_creates_parent_and_writes_stable_record(tmp_path) -> None:
    path = tmp_path / "nested" / "events.jsonl"
    writer = JsonlEventWriter(path)

    writer.write(
        RuntimeEvent(
            event_type="observation",
            at_ns=1_234,
            payload={"confidence": 0.95, "node": "B"},
        )
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {
        "at_ns": 1_234,
        "event_type": "observation",
        "payload": {"confidence": 0.95, "node": "B"},
        "schema_version": 1,
    }


def test_jsonl_event_writer_appends_without_overwriting(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    writer = JsonlEventWriter(path)

    writer.write(RuntimeEvent(event_type="start", at_ns=1, payload={}))
    writer.write(RuntimeEvent(event_type="stop", at_ns=2, payload={"reason": "done"}))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event_type"] for record in records] == ["start", "stop"]


def test_jsonl_event_writer_binds_run_id_and_can_truncate_old_run(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"run_id":"old-run"}\n', encoding="utf-8")
    writer = JsonlEventWriter(path, run_id="new-run", truncate=True)

    writer.write(RuntimeEvent(event_type="start", at_ns=1, payload={}))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["run_id"] == "new-run"
    assert "old-run" not in path.read_text(encoding="utf-8")


def test_runtime_event_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="event_type"):
        RuntimeEvent(event_type="", at_ns=-1, payload={})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_jsonl_event_writer_rejects_non_finite_numbers(tmp_path, value: float) -> None:
    writer = JsonlEventWriter(tmp_path / "events.jsonl")

    with pytest.raises(ValueError, match="JSON"):
        writer.write(RuntimeEvent(event_type="metric", at_ns=1, payload={"value": value}))
