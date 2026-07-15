"""截图帧模型与确定性离线回放。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    sequence: int
    captured_at_ns: int
    image: NDArray[np.uint8]
    source: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


class FrameRecorder:
    """先落 PNG、再追加 manifest，避免清单引用未完成图像。"""

    def __init__(self, directory: str | Path, *, image_writer=cv2.imwrite) -> None:
        self._directory = Path(directory)
        self._frames_directory = self._directory / "frames"
        self._manifest = self._directory / "manifest.jsonl"
        self._frames_directory.mkdir(parents=True, exist_ok=True)
        self._image_writer = image_writer
        self._previous_sequence = -1
        self._previous_captured_at_ns = -1

    def record(
        self,
        frame: CapturedFrame,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        if (
            frame.sequence <= self._previous_sequence
            or frame.captured_at_ns <= self._previous_captured_at_ns
        ):
            raise ValueError("录制帧序号和时间戳必须单调递增")
        if (
            frame.image.dtype != np.uint8
            or frame.image.ndim != 3
            or frame.image.shape[2] != 3
        ):
            raise ValueError("录制帧必须是 H×W×3 的 uint8 BGR 图像")

        relative_path = Path("frames") / f"frame-{frame.sequence:08d}.png"
        final_path = self._directory / relative_path
        temporary_path = final_path.with_name(f".{final_path.stem}.tmp.png")
        record = {
            "sequence": frame.sequence,
            "captured_at_ns": frame.captured_at_ns,
            "image": relative_path.as_posix(),
            "source": frame.source,
            "width": int(frame.image.shape[1]),
            "height": int(frame.image.shape[0]),
            "metadata": dict(metadata or frame.metadata),
        }
        serialized = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if not self._image_writer(str(temporary_path), frame.image):
            raise OSError(f"写入截图失败: {temporary_path}")
        temporary_path.replace(final_path)
        with self._manifest.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.write("\n")
        self._previous_sequence = frame.sequence
        self._previous_captured_at_ns = frame.captured_at_ns
        return final_path


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
                expected_width = int(record.get("width", image.shape[1]))
                expected_height = int(record.get("height", image.shape[0]))
                if image.shape[:2] != (expected_height, expected_width):
                    raise ValueError(
                        f"第 {line_number} 行分辨率与图像不一致: "
                        f"manifest={expected_width}x{expected_height}, "
                        f"image={image.shape[1]}x{image.shape[0]}"
                    )
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
                    metadata=record.get("metadata", {}),
                )
