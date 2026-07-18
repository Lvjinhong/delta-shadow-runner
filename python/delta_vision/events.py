"""可追溯的 JSONL 运行事件。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .jsonl_io import (
    append_jsonl_record,
    load_jsonl_records,
    repair_trailing_partial_record,
)


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
        self._idempotency_keys: set[str] = set()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            self._path.write_text("", encoding="utf-8")
        self._load_idempotency_keys()

    def _load_idempotency_keys(self) -> None:
        self._idempotency_keys.clear()
        for record in load_jsonl_records(self._path):
            idempotency_key = record.get("idempotency_key")
            if isinstance(idempotency_key, str) and idempotency_key:
                self._idempotency_keys.add(idempotency_key)

    def write(
        self,
        event: RuntimeEvent,
        *,
        idempotency_key: str | None = None,
    ) -> None:
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str) or not idempotency_key
        ):
            raise ValueError("idempotency_key 必须是非空字符串")
        if idempotency_key is not None and idempotency_key in self._idempotency_keys:
            return
        record = {
            "schema_version": 1,
            "event_type": event.event_type,
            "at_ns": event.at_ns,
            "payload": dict(event.payload),
        }
        if self._run_id is not None:
            record["run_id"] = self._run_id
        if idempotency_key is not None:
            record["idempotency_key"] = idempotency_key
        serialized = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            append_jsonl_record(self._path, serialized)
        except BaseException as error:
            # close/flush 可能在完整追加后才报错；重新读取一次磁盘状态，供旧游标重试。
            try:
                repair_trailing_partial_record(self._path)
                self._load_idempotency_keys()
            except BaseException as recovery_error:
                error.add_note(
                    "恢复 runtime JSONL 时失败: "
                    f"{type(recovery_error).__name__}: {recovery_error}"
                )
            raise
        if idempotency_key is not None:
            self._idempotency_keys.add(idempotency_key)
