"""运行证据的通用原子写入工具。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def write_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """同目录写临时文件后替换，避免留下半份 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
