"""可追溯的 JSONL 运行事件。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event_type: str
    at_ns: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event_type 不能为空")
        if self.at_ns < 0:
            raise ValueError("at_ns 不能为负数")


class JsonlEventWriter:
    """每条事件立即追加落盘，进程异常时也保留已完成记录。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: RuntimeEvent) -> None:
        record = {
            "schema_version": 1,
            "event_type": event.event_type,
            "at_ns": event.at_ns,
            "payload": dict(event.payload),
        }
        serialized = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.write("\n")
