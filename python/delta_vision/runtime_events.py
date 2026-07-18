"""在导航与菜单 Worker 之间共享输入事件持久化契约。"""

from __future__ import annotations

from typing import Protocol

from .events import JsonlEventWriter, RuntimeEvent
from .frames import FrameRecorder


class InputEventSource(Protocol):
    @property
    def events(self) -> tuple[object, ...]: ...


def input_event_payload(event: object, *, event_sequence: int) -> dict[str, object]:
    kind = getattr(event, "kind", None)
    at_ns = getattr(event, "at_ns", None)
    if not isinstance(kind, str) or not kind:
        raise ValueError("输入事件 kind 必须是非空字符串")
    if type(at_ns) is not int or at_ns < 0:
        raise ValueError("输入事件 at_ns 必须是非负整数")
    event_id = f"input:{event_sequence}:{at_ns}:{kind}"
    return {
        "event_id": event_id,
        "kind": kind,
        "at_ns": at_ns,
        "key": getattr(event, "key", None),
        "dx": getattr(event, "dx", None),
        "dy": getattr(event, "dy", None),
        "x": getattr(event, "x", None),
        "y": getattr(event, "y", None),
        "reason": getattr(event, "reason", None),
    }


def persist_new_input_events(
    *,
    actuator: InputEventSource,
    input_event_cursor: int,
    recorder: FrameRecorder,
    event_writer: JsonlEventWriter,
) -> tuple[int, list[dict[str, object]]]:
    """从稳定游标开始持久化新事件，并保留绝对与相对鼠标坐标。"""

    current_input_events = actuator.events
    if (
        type(input_event_cursor) is not int
        or input_event_cursor < 0
        or input_event_cursor > len(current_input_events)
    ):
        raise ValueError("输入事件游标超出当前事件流")
    new_input_events = current_input_events[input_event_cursor:]
    input_payloads = [
        input_event_payload(event, event_sequence=event_sequence)
        for event_sequence, event in enumerate(
            new_input_events,
            start=input_event_cursor,
        )
    ]
    for input_payload in input_payloads:
        at_ns = input_payload["at_ns"]
        if type(at_ns) is not int:
            raise AssertionError("已校验的输入事件 at_ns 类型发生变化")
        event_id = input_payload["event_id"]
        if not isinstance(event_id, str):
            raise AssertionError("已生成的输入事件 ID 类型发生变化")
        recorder.record_input_event(
            at_ns=at_ns,
            payload=input_payload,
            event_id=event_id,
        )
        event_writer.write(
            RuntimeEvent(
                event_type="input",
                at_ns=at_ns,
                payload=input_payload,
            ),
            idempotency_key=event_id,
        )
    return len(current_input_events), input_payloads
