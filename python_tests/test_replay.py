import hashlib
import json

import cv2
import numpy as np
import pytest

from delta_vision.frames import CapturedFrame, FrameRecorder, ReplayFrameSource


def _write_replay(directory) -> None:
    frames = [
        np.full((12, 16, 3), fill_value=17, dtype=np.uint8),
        np.full((12, 16, 3), fill_value=29, dtype=np.uint8),
    ]
    manifest = directory / "manifest.jsonl"
    records = []
    for sequence, frame in enumerate(frames):
        name = f"frame-{sequence:06d}.png"
        assert cv2.imwrite(str(directory / name), frame)
        records.append(
            {
                "sequence": sequence,
                "captured_at_ns": 1_000_000_000 + sequence * 50_000_000,
                "image": name,
                "source": "fixture",
            }
        )
    manifest.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")


def test_replay_preserves_manifest_order_metadata_and_pixels(tmp_path) -> None:
    _write_replay(tmp_path)

    frames = list(ReplayFrameSource(tmp_path))

    assert [frame.sequence for frame in frames] == [0, 1]
    assert [frame.captured_at_ns for frame in frames] == [1_000_000_000, 1_050_000_000]
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [17, 29]
    assert all(frame.image.flags.writeable is False for frame in frames)


def test_replay_is_deterministic_across_multiple_iterations(tmp_path) -> None:
    _write_replay(tmp_path)
    source = ReplayFrameSource(tmp_path)

    digests = []
    for _ in range(2):
        digest = hashlib.sha256()
        for frame in source:
            digest.update(frame.sequence.to_bytes(8, "little"))
            digest.update(frame.captured_at_ns.to_bytes(8, "little"))
            digest.update(frame.image.tobytes())
        digests.append(digest.hexdigest())

    assert digests[0] == digests[1]


def test_replay_rejects_missing_image(tmp_path) -> None:
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sequence": 0,
                "captured_at_ns": 1,
                "image": "missing.png",
                "source": "fixture",
            }
        ),
        encoding="utf-8",
    )

    source = ReplayFrameSource(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"missing\.png"):
        list(source)


def test_replay_rejects_timestamp_going_backwards(tmp_path) -> None:
    _write_replay(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records[1]["captured_at_ns"] = records[0]["captured_at_ns"] - 1
    manifest.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")

    with pytest.raises(ValueError, match="时间戳必须单调递增"):
        list(ReplayFrameSource(tmp_path))


def test_replay_rejects_corrupt_image(tmp_path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not an image")
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sequence": 0,
                "captured_at_ns": 1,
                "image": "broken.png",
                "source": "fixture",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="无法解码"):
        list(ReplayFrameSource(tmp_path))


def test_replay_rejects_image_outside_replay_directory(tmp_path) -> None:
    replay = tmp_path / "replay"
    replay.mkdir()
    outside = tmp_path / "outside.png"
    assert cv2.imwrite(str(outside), np.zeros((4, 4, 3), dtype=np.uint8))
    (replay / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sequence": 0,
                "captured_at_ns": 1,
                "image": str(outside),
                "source": "fixture",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="必须位于回放目录内"):
        list(ReplayFrameSource(replay))


def test_frame_recorder_round_trips_resolution_and_action_metadata(tmp_path) -> None:
    recorder = FrameRecorder(tmp_path)
    first = np.full((12, 16, 3), 17, dtype=np.uint8)
    second = np.full((12, 16, 3), 29, dtype=np.uint8)

    recorder.record(
        CapturedFrame(0, 1_000, first, "dxcam:0:0"),
        metadata={"action": {"kind": "key_down", "key": "w"}},
    )
    recorder.record(
        CapturedFrame(1, 2_000, second, "dxcam:0:0"),
        metadata={"action": {"kind": "key_up", "key": "w"}},
    )

    replayed = list(ReplayFrameSource(tmp_path))
    records = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(record["width"], record["height"]) for record in records] == [
        (16, 12),
        (16, 12),
    ]
    assert replayed[0].metadata == {
        "action": {"kind": "key_down", "key": "w"}
    }
    assert replayed[1].metadata == {"action": {"kind": "key_up", "key": "w"}}


def test_frame_recorder_rejects_non_monotonic_frames(tmp_path) -> None:
    recorder = FrameRecorder(tmp_path)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    recorder.record(CapturedFrame(1, 100, image, "fixture"))

    with pytest.raises(ValueError, match="单调递增"):
        recorder.record(CapturedFrame(1, 101, image, "fixture"))
    with pytest.raises(ValueError, match="单调递增"):
        recorder.record(CapturedFrame(2, 99, image, "fixture"))


def test_frame_recorder_does_not_append_manifest_when_image_write_fails(tmp_path) -> None:
    recorder = FrameRecorder(tmp_path, image_writer=lambda *_: False)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(OSError, match="写入截图失败"):
        recorder.record(CapturedFrame(0, 100, image, "fixture"))

    manifest = tmp_path / "manifest.jsonl"
    assert not manifest.exists() or manifest.read_text(encoding="utf-8") == ""


def test_replay_rejects_manifest_resolution_mismatch(tmp_path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "frame.png"), image)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sequence": 0,
                "captured_at_ns": 1,
                "image": "frame.png",
                "source": "fixture",
                "width": 999,
                "height": 4,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="分辨率"):
        list(ReplayFrameSource(tmp_path))
