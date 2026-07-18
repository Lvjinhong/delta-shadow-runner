"""JSONL 追加失败后的最小修复与严格读取。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def append_jsonl_record(path: Path, serialized: str) -> None:
    """以一次文本写调用追加完整记录；调用方仍需处理底层半写异常。"""

    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{serialized}\n")


def repair_trailing_partial_record(path: Path) -> bool:
    """只移除没有换行终止的尾部半条记录，不掩盖中间损坏。"""

    if not path.is_file():
        return False
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return False
    last_complete_end = payload.rfind(b"\n") + 1
    with path.open("r+b") as stream:
        stream.truncate(last_complete_end)
    return True


def load_jsonl_records(path: Path) -> tuple[Mapping[str, object], ...]:
    """严格读取完整 JSONL；任何中间坏行都阻止继续运行。"""

    if not path.is_file():
        return ()
    payload = path.read_bytes()
    if not payload:
        return ()
    if not payload.endswith(b"\n"):
        raise ValueError(f"JSONL 存在未完成的尾部记录: {path}")
    records: list[Mapping[str, object]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"JSONL 第 {line_number} 行损坏: {path}") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"JSONL 第 {line_number} 行必须是对象: {path}")
        records.append(record)
    return tuple(records)
