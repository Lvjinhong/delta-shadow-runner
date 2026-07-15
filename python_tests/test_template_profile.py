import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from delta_vision.frames import CapturedFrame, DatasetContentDigest
from delta_vision.template_profile import load_template_profile


def _template(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(12, 16, 3), dtype=np.uint8)


def _write_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_digest(frame_hashes: list[tuple[int, str]]) -> str:
    digest = DatasetContentDigest()
    for sequence, frame_sha256 in frame_hashes:
        digest.update_hash(sequence, frame_sha256)
    return digest.hexdigest()


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    first_hash = _write_image(tmp_path / "templates" / "first.png", _template(1))
    second_hash = _write_image(tmp_path / "templates" / "second.png", _template(2))
    manifest = {
        "schema_version": 2,
        "capture_profile": {"width": 180, "height": 120},
        "matcher": {
            "scales": [1.0],
            "score_threshold": 0.8,
            "minimum_spatial_margin": 0.05,
            "minimum_template_margin": 0.05,
            "nms_radius_px": 18,
        },
        "rois": {"scene": {"left": 20, "top": 10, "width": 120, "height": 80}},
        "source_datasets": [
            {
                "run_id": "game-run-001",
                "frame_sha256s": ["1" * 64, "2" * 64],
                "frame_hashes": [
                    {"sequence": 10, "sha256": "1" * 64},
                    {"sequence": 20, "sha256": "2" * 64},
                ],
                "perception_sha256s": ["a" * 64, "b" * 64],
                "dataset_content_sha256": _dataset_digest(
                    [(10, "1" * 64), (20, "2" * 64)]
                ),
                "run_json_sha256": "c" * 64,
                "frame_manifest_sha256": "d" * 64,
            }
        ],
        "templates": [
            {
                "id": "route-001",
                "image": "templates/first.png",
                "sha256": first_hash,
                "roi_id": "scene",
                "route_position": [200.0, 10.0],
                "waypoint_id": "turn",
                "source_run_id": "game-run-001",
                "source_sequence": 10,
                "source_frame_sha256": "1" * 64,
            },
            {
                "id": "route-002",
                "image": "templates/second.png",
                "sha256": second_hash,
                "roi_id": "scene",
                "route_position": [300.0, 20.0],
                "waypoint_id": None,
                "source_run_id": "game-run-001",
                "source_sequence": 20,
                "source_frame_sha256": "2" * 64,
            },
        ],
    }
    path = tmp_path / "templates.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _write_manifest(path: Path, manifest: object) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_template_profile_builds_traceable_route_observer(tmp_path) -> None:
    path, _ = _manifest(tmp_path)
    profile = load_template_profile(path)
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[35:47, 60:76] = _template(1)
    frame.setflags(write=False)

    observation = profile.observer.observe(CapturedFrame(1, 2, frame, "fixture"))

    assert profile.frame_size == (180, 120)
    assert len(profile.manifest_sha256) == 64
    assert profile.source_run_ids == frozenset({"game-run-001"})
    assert profile.source_frame_sha256s == frozenset({"1" * 64, "2" * 64})
    assert profile.source_perception_sha256s == frozenset({"a" * 64, "b" * 64})
    assert len(profile.perception_regions) == 1
    assert observation.centroid == (200.0, 10.0)
    assert observation.waypoint_id == "turn"


def test_load_template_profile_rejects_hash_mismatch(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["templates"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_template_profile(path)


def test_load_template_profile_binds_source_frame_hash_to_same_run(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["source_datasets"].append(
        {
            "run_id": "game-run-002",
            "frame_sha256s": ["3" * 64],
            "frame_hashes": [{"sequence": 10, "sha256": "3" * 64}],
            "perception_sha256s": ["e" * 64],
            "dataset_content_sha256": _dataset_digest([(10, "3" * 64)]),
            "run_json_sha256": "f" * 64,
            "frame_manifest_sha256": "0" * 64,
        }
    )
    manifest["templates"][0]["source_frame_sha256"] = "3" * 64
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="同一 source dataset"):
        load_template_profile(path)


def test_load_template_profile_binds_source_hash_to_exact_sequence(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["templates"][0]["source_frame_sha256"] = "2" * 64
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match=r"source_sequence.*source_frame_sha256"):
        load_template_profile(path)


def test_load_template_profile_accepts_uppercase_hash_and_reports_exact_manifest_hash(
    tmp_path,
) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["templates"][0]["sha256"] = manifest["templates"][0]["sha256"].upper()
    _write_manifest(path, manifest)

    profile = load_template_profile(path)

    assert profile.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.png",
        "..\\outside.png",
        "/tmp/outside.png",
        "C:\\outside.png",
        "C:outside.png",
        "\\\\host\\x.png",
        "templates\\first.png:stream",
    ],
)
def test_load_template_profile_rejects_unsafe_image_path(tmp_path, unsafe_path: str) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["templates"][0]["image"] = unsafe_path
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="模板目录"):
        load_template_profile(path)


def test_load_template_profile_rejects_symlink_escape(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    outside = tmp_path.parent / "outside-template.png"
    _write_image(outside, _template(3))
    link = tmp_path / "templates" / "link.png"
    link.symlink_to(outside)
    manifest["templates"][0]["image"] = "templates/link.png"
    manifest["templates"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="模板目录"):
        load_template_profile(path)


@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (lambda value: value.update(capture_profile=None), "capture_profile"),
        (lambda value: value.update(schema_version=3), "schema_version"),
        (
            lambda value: value["capture_profile"].update(width=0),
            "capture_profile.width",
        ),
        (
            lambda value: value["rois"]["scene"].update(left=-1),
            "ROI",
        ),
        (
            lambda value: value["rois"]["scene"].update(width=1000),
            "capture_profile",
        ),
        (lambda value: value.update(rois={}), "至少需要一个 ROI"),
        (
            lambda value: value.update(rois={"": {"left": 0, "top": 0, "width": 1, "height": 1}}),
            "ROI ID",
        ),
        (
            lambda value: value["rois"]["scene"].update(width="120"),
            "必须是整数",
        ),
        (
            lambda value: value["rois"]["scene"].update(left=True),
            "必须是整数",
        ),
        (
            lambda value: value["rois"]["scene"].update(width=0),
            "宽高必须为正数",
        ),
        (lambda value: value["matcher"].update(scales=[]), "scales"),
        (
            lambda value: value["matcher"].update(scales=[float("nan")]),
            "scales",
        ),
        (lambda value: value["matcher"].update(scales=[0]), "scales"),
        (lambda value: value.update(templates=[]), "templates"),
        (lambda value: value.update(source_datasets=[]), "source_datasets"),
        (
            lambda value: value["source_datasets"][0].update(frame_sha256s=["g" * 64]),
            "frame_sha256s",
        ),
        (
            lambda value: value["source_datasets"][0].update(perception_sha256s=["g" * 64]),
            "perception_sha256s",
        ),
        (
            lambda value: value["source_datasets"][0].update(frame_manifest_sha256="g" * 64),
            "frame_manifest_sha256",
        ),
        (
            lambda value: value["source_datasets"][0].update(
                dataset_content_sha256="f" * 64
            ),
            "dataset_content_sha256",
        ),
        (lambda value: value["templates"][0].update(id=""), "templates\\[0\\].id"),
        (
            lambda value: value["templates"][0].update(roi_id="missing"),
            "roi_id",
        ),
        (
            lambda value: value["templates"][0].update(route_position=[0]),
            "route_position",
        ),
        (
            lambda value: value["templates"][0].update(route_position=[float("inf"), 0]),
            "route_position",
        ),
        (
            lambda value: value["templates"][0].update(waypoint_id=""),
            "waypoint_id",
        ),
        (
            lambda value: value["templates"][0].update(source_run_id=""),
            "source_run_id",
        ),
        (
            lambda value: value["templates"][0].update(source_run_id="other-run"),
            "source_datasets",
        ),
        (
            lambda value: value["templates"][0].update(source_sequence=-1),
            "source_sequence",
        ),
        (
            lambda value: value["templates"][0].update(source_frame_sha256="g" * 64),
            "source_frame_sha256",
        ),
        (
            lambda value: value["templates"][0].update(source_frame_sha256="3" * 64),
            "同一 source dataset",
        ),
        (
            lambda value: value["templates"][1].update(id="route-001"),
            "不能重复",
        ),
        (
            lambda value: value["templates"][0].update(image=""),
            "非空相对路径",
        ),
        (
            lambda value: value["templates"][0].update(sha256="g" * 64),
            "64 位十六进制",
        ),
    ],
)
def test_load_template_profile_rejects_invalid_contract(tmp_path, mutate, error_match: str) -> None:
    path, manifest = _manifest(tmp_path)
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_template_profile(path)


def test_load_template_profile_rejects_non_object_root(tmp_path) -> None:
    path = tmp_path / "templates.json"
    _write_manifest(path, [])

    with pytest.raises(ValueError, match="root"):
        load_template_profile(path)


@pytest.mark.parametrize("schema_version", [True, 2.0, "2"])
def test_load_template_profile_requires_exact_integer_schema_version(
    tmp_path, schema_version: object
) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["schema_version"] = schema_version
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="schema_version"):
        load_template_profile(path)


def test_load_template_profile_reports_v1_recalibration_migration(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["schema_version"] = 1
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match=r"schema_version=1.*重新标定"):
        load_template_profile(path)


def test_load_template_profile_rejects_non_monotonic_frame_hash_sequences(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    dataset = manifest["source_datasets"][0]
    dataset["frame_hashes"].reverse()
    dataset["frame_sha256s"].reverse()
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="单调递增"):
        load_template_profile(path)


def test_load_template_profile_rejects_missing_image(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["templates"][0]["image"] = "templates/missing.png"
    _write_manifest(path, manifest)

    with pytest.raises(FileNotFoundError, match="模板图像不存在"):
        load_template_profile(path)


def test_load_template_profile_rejects_undecodable_image(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    broken = tmp_path / "templates" / "broken.png"
    broken.write_bytes(b"not-an-image")
    manifest["templates"][0]["image"] = "templates/broken.png"
    manifest["templates"][0]["sha256"] = hashlib.sha256(broken.read_bytes()).hexdigest()
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="无法解码"):
        load_template_profile(path)


def test_loaded_template_is_bound_to_the_hashed_image_bytes(tmp_path, monkeypatch) -> None:
    path, _ = _manifest(tmp_path)
    original = _template(1)
    replacement = _template(99)
    image_path = tmp_path / "templates" / "first.png"
    real_imread = cv2.imread

    def replace_before_imread(filename, flags):
        _write_image(image_path, replacement)
        return real_imread(filename, flags)

    monkeypatch.setattr(cv2, "imread", replace_before_imread)
    profile = load_template_profile(path)
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[35:47, 60:76] = original
    frame.setflags(write=False)

    observation = profile.observer.observe(CapturedFrame(1, 2, frame, "fixture"))

    assert observation.centroid == (200.0, 10.0)


def test_load_template_profile_accepts_roi_and_template_exact_boundaries(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["rois"]["scene"] = {"left": 0, "top": 0, "width": 180, "height": 120}
    _write_manifest(path, manifest)

    profile = load_template_profile(path)

    assert profile.frame_size == (180, 120)


def test_load_template_profile_rejects_template_that_never_fits_roi(tmp_path) -> None:
    path, manifest = _manifest(tmp_path)
    oversized = _template(4)
    oversized = cv2.resize(oversized, (240, 160), interpolation=cv2.INTER_NEAREST)
    oversized_path = tmp_path / "templates" / "oversized.png"
    manifest["templates"][0]["sha256"] = _write_image(oversized_path, oversized)
    manifest["templates"][0]["image"] = "templates/oversized.png"
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="没有可用缩放比例"):
        load_template_profile(path)


def test_load_template_profile_rejects_scale_that_degenerates_to_one_pixel(
    tmp_path,
) -> None:
    path, manifest = _manifest(tmp_path)
    manifest["matcher"]["scales"] = [0.01]
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="没有可用缩放比例"):
        load_template_profile(path)
