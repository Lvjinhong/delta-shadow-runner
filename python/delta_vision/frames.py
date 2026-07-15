"""截图帧模型与确定性离线回放。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    sequence: int
    captured_at_ns: int
    image: NDArray[np.uint8]
    source: str


class ReplayFrameSource:
    """每次迭代都从 manifest 重新读取，保证回放不共享游标。"""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._manifest = self._directory / "manifest.jsonl"

    def __iter__(self) -> Iterator[CapturedFrame]:
        if not self._manifest.is_file():
            raise FileNotFoundError(f"回放清单不存在: {self._manifest}")

        previous_sequence = -1
        previous_captured_at_ns = -1
        replay_root = self._directory.resolve()
        with self._manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                image_path = (replay_root / str(record["image"])).resolve()
                try:
                    image_path.relative_to(replay_root)
                except ValueError as error:
                    raise ValueError(
                        f"第 {line_number} 行图像必须位于回放目录内: {image_path}"
                    ) from error
                if not image_path.is_file():
                    raise FileNotFoundError(f"回放图像不存在: {image_path}")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"无法解码第 {line_number} 行图像: {image_path}")
                sequence = int(record["sequence"])
                captured_at_ns = int(record["captured_at_ns"])
                if sequence <= previous_sequence:
                    raise ValueError(f"第 {line_number} 行序号必须单调递增")
                if captured_at_ns <= previous_captured_at_ns:
                    raise ValueError(f"第 {line_number} 行时间戳必须单调递增")
                previous_sequence = sequence
                previous_captured_at_ns = captured_at_ns
                image.setflags(write=False)
                yield CapturedFrame(
                    sequence=sequence,
                    captured_at_ns=captured_at_ns,
                    image=image,
                    source=str(record["source"]),
                )
