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

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        truncate: bool = False,
    ) -> None:
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("run_id 必须是非空字符串")
        self._path = Path(path)
        self._run_id = run_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            self._path.write_text("", encoding="utf-8")

    def write(self, event: RuntimeEvent) -> None:
        record = {
            "schema_version": 1,
            "event_type": event.event_type,
            "at_ns": event.at_ns,
            "payload": dict(event.payload),
        }
        if self._run_id is not None:
            record["run_id"] = self._run_id
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
