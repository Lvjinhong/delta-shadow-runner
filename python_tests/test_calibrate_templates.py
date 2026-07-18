import gc
import hashlib
import json
import math
import weakref

import cv2
import numpy as np
import pytest

from delta_vision.calibrate_templates import (
    FeatureMatcherConfiguration,
    MatcherConfiguration,
    NormalizedRoi,
    calibrate_templates,
    main,
)
from delta_vision.feature_matching import FeatureBackend
from delta_vision.frames import (
    CapturedFrame,
    DatasetContentDigest,
    FrameRecorder,
    frame_content_sha256,
)
from delta_vision.navigation import ObservationScope
from delta_vision.template_profile import load_template_profile


def _frame(sequence: int, seed: int, *, width: int = 100, height: int = 80) -> CapturedFrame:
    image = np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    image.setflags(write=False)
    return CapturedFrame(
        sequence,
        1_000 + sequence,
        image,
        "fixture",
        {
            "run_id": "game-run-001",
            "dataset_kind": "manual-game-route",
            "dataset_split": "calibration",
        },
    )


def _dataset(tmp_path, frames: list[CapturedFrame] | None = None):
    dataset = tmp_path / "dataset"
    recorder = FrameRecorder(dataset)
    actual_frames = frames if frames is not None else [_frame(0, 1), _frame(1, 2)]
    for frame in actual_frames:
        recorder.record(frame)
    (dataset / "run.json").write_text(
        json.dumps(
            {
                "run_id": "game-run-001",
                "window_title": "三角洲行动",
                "backend": "dxcam",
                "dataset_split": "calibration",
                "frame_count": len(actual_frames),
                "resolution": [100, 80],
            }
        ),
        encoding="utf-8",
    )
    return dataset, actual_frames


def _label(
    *,
    sequence: int = 0,
    template_id: str = "spawn",
    split: str = "calibration",
    run_id: str = "game-run-001",
    roi=None,
):
    return {
        "run_id": run_id,
        "sequence": sequence,
        "split": split,
        "locatable": True,
        "template_id": template_id,
        "roi": roi
        or {
            "id": "scene",
            "x": 0.25,
            "y": 0.25,
            "width": 0.5,
            "height": 0.5,
        },
        "route_position": [sequence * 100.0, 0.0],
        "waypoint_id": template_id,
    }


def _write_labels(path, labels) -> None:
    path.write_text(
        "".join(json.dumps(label, allow_nan=False) + "\n" for label in labels),
        encoding="utf-8",
    )


def test_normalized_roi_uses_floor_for_origin_and_ceil_for_far_edge() -> None:
    region = NormalizedRoi(0.101, 0.201, 0.302, 0.402).to_region(100, 100)

    assert (region.left, region.top, region.right, region.bottom) == (10, 20, 41, 61)
    full = NormalizedRoi(0, 0, 1, 1).to_region(100, 80)
    assert (full.left, full.top, full.width, full.height) == (0, 0, 100, 80)


@pytest.mark.parametrize(
    ("x", "y", "width", "height"),
    [
        (-0.1, 0, 0.5, 0.5),
        (0, 0, 0, 0.5),
        (0.8, 0, 0.3, 0.5),
        (0, 0.8, 0.5, 0.3),
        (math.nan, 0, 0.5, 0.5),
        (0, math.inf, 0.5, 0.5),
        (True, 0, 0.5, 0.5),
    ],
)
def test_normalized_roi_rejects_invalid_contract(x, y, width, height) -> None:
    with pytest.raises(ValueError):
        NormalizedRoi(x, y, width, height)


@pytest.mark.parametrize(("frame_width", "frame_height"), [(0, 80), (100, True)])
def test_normalized_roi_rejects_invalid_frame_size(frame_width, frame_height) -> None:
    with pytest.raises(ValueError):
        NormalizedRoi(0, 0, 1, 1).to_region(frame_width, frame_height)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scales", ()),
        ("scales", (0,)),
        ("scales", (math.nan,)),
        ("scales", ("bad",)),
        ("score_threshold", 0),
        ("score_threshold", math.nan),
        ("score_threshold", "bad"),
        ("minimum_spatial_margin", -0.1),
        ("minimum_spatial_margin", "bad"),
        ("minimum_template_margin", 1.1),
        ("minimum_template_margin", "bad"),
        ("nms_radius_px", 0),
        ("nms_radius_px", True),
    ],
)
def test_matcher_configuration_rejects_invalid_contract(field, value) -> None:
    values = {
        "scales": (1.0,),
        "score_threshold": 0.82,
        "minimum_spatial_margin": 0.05,
        "minimum_template_margin": 0.05,
        "nms_radius_px": 18,
    }
    values[field] = value

    with pytest.raises(ValueError):
        MatcherConfiguration(**values)


def test_feature_matcher_configuration_has_traceable_backend_defaults() -> None:
    configuration = FeatureMatcherConfiguration.default(FeatureBackend.SIFT)

    assert configuration.backend is FeatureBackend.SIFT
    assert configuration.maximum_features == 3000
    assert configuration.policy.minimum_inliers >= 4
    assert configuration.to_manifest()["backend"] == "sift"


def test_calibrate_templates_can_emit_loadable_sift_profile(tmp_path) -> None:
    dataset, frames = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(
        labels_path,
        [
            _label(
                roi={
                    "id": "scene",
                    "x": 0,
                    "y": 0,
                    "width": 1,
                    "height": 1,
                }
            )
        ],
    )

    result = calibrate_templates(
        dataset_directory=dataset,
        labels_path=labels_path,
        output_directory=tmp_path / "sift-profile",
        matcher=MatcherConfiguration.default(),
        feature_matcher=FeatureMatcherConfiguration.default(FeatureBackend.SIFT),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    profile = load_template_profile(result.manifest_path)
    observation = profile.observer.observe(
        frames[0],
        scope=ObservationScope(allowed_waypoint_ids=None),
    )

    assert manifest["feature_matcher"]["backend"] == "sift"
    assert observation.waypoint_id == "spawn"


def test_feature_calibration_rejects_inconsistent_waypoint_positions_before_writing(
    tmp_path,
) -> None:
    dataset, _frames = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    full_roi = {"id": "scene", "x": 0, "y": 0, "width": 1, "height": 1}
    first = _label(sequence=0, template_id="spawn-day", roi=full_roi)
    first["waypoint_id"] = "spawn"
    first["route_position"] = [0.0, 0.0]
    second = _label(sequence=1, template_id="spawn-night", roi=full_roi)
    second["waypoint_id"] = "spawn"
    second["route_position"] = [10.0, 0.0]
    _write_labels(labels_path, [first, second])
    output = tmp_path / "sift-profile"

    with pytest.raises(ValueError, match=r"同一 waypoint.*路线坐标"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=output,
            matcher=MatcherConfiguration.default(),
            feature_matcher=FeatureMatcherConfiguration.default(FeatureBackend.SIFT),
        )

    assert not output.exists()


def test_calibrate_templates_crops_exact_pixels_and_writes_traceable_profile(
    tmp_path,
) -> None:
    dataset, frames = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label(sequence=0), _label(sequence=1, template_id="turn")])

    result = calibrate_templates(
        dataset_directory=dataset,
        labels_path=labels_path,
        output_directory=tmp_path / "profile",
        matcher=MatcherConfiguration.default(),
    )

    assert result.template_count == 2
    assert result.run_id == "game-run-001"
    manifest_path = tmp_path / "profile" / "templates.json"
    profile = load_template_profile(manifest_path)
    assert profile.frame_size == (100, 80)
    assert profile.source_run_ids == frozenset({"game-run-001"})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["rois"]["scene"] == {
        "left": 25,
        "top": 20,
        "width": 50,
        "height": 40,
    }
    template_path = tmp_path / "profile" / manifest["templates"][0]["image"]
    decoded = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, frames[0].image[20:60, 25:75])
    assert (
        manifest["templates"][0]["sha256"] == hashlib.sha256(template_path.read_bytes()).hexdigest()
    )
    assert manifest["templates"][0]["source_sequence"] == 0
    assert manifest["templates"][0]["source_frame_sha256"] == frame_content_sha256(frames[0].image)
    assert manifest["source_datasets"] == [
        {
            "run_id": "game-run-001",
            "frame_sha256s": [
                frame_content_sha256(frames[0].image),
                frame_content_sha256(frames[1].image),
            ],
            "frame_hashes": [
                {
                    "sequence": 0,
                    "sha256": frame_content_sha256(frames[0].image),
                },
                {
                    "sequence": 1,
                    "sha256": frame_content_sha256(frames[1].image),
                },
            ],
            "perception_sha256s": [
                frame_content_sha256(frames[0].image[20:60, 25:75]),
                frame_content_sha256(frames[1].image[20:60, 25:75]),
            ],
            "dataset_content_sha256": _dataset_content_sha256(frames),
            "run_json_sha256": hashlib.sha256((dataset / "run.json").read_bytes()).hexdigest(),
            "frame_manifest_sha256": hashlib.sha256(
                (dataset / "manifest.jsonl").read_bytes()
            ).hexdigest(),
        }
    ]


def _dataset_content_sha256(frames: list[CapturedFrame]) -> str:
    digest = DatasetContentDigest()
    for frame in frames:
        digest.update(frame.sequence, frame.image)
    return digest.hexdigest()


def test_calibrate_templates_streams_frames_without_retaining_history(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.jsonl").write_text("fixture\n", encoding="utf-8")
    (dataset / "run.json").write_text(
        json.dumps(
            {
                "run_id": "game-run-001",
                "dataset_split": "calibration",
                "frame_count": 3,
                "resolution": [100, 80],
            }
        ),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label(sequence=0)])

    class MemoryCheckingSource:
        def __init__(self, _directory) -> None:
            self.sequence = 0
            self.references = []

        def __iter__(self):
            return self

        def __next__(self):
            if self.sequence >= 3:
                raise StopIteration
            if self.sequence >= 2:
                gc.collect()
                assert self.references[self.sequence - 2]() is None
            frame = _frame(self.sequence, self.sequence + 1)
            self.references.append(weakref.ref(frame.image))
            self.sequence += 1
            return frame

    monkeypatch.setattr(
        "delta_vision.calibrate_templates.ReplayFrameSource",
        MemoryCheckingSource,
    )

    result = calibrate_templates(
        dataset_directory=dataset,
        labels_path=labels_path,
        output_directory=tmp_path / "profile",
        matcher=MatcherConfiguration.default(),
    )

    assert result.template_count == 1


def test_calibrate_templates_requires_calibration_dataset_split(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path)
    run_path = dataset / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["dataset_split"] = "blind"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match="dataset_split"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_requires_calibration_split_on_every_frame(
    tmp_path,
) -> None:
    frame = _frame(0, 1)
    poisoned = CapturedFrame(
        frame.sequence,
        frame.captured_at_ns,
        frame.image,
        frame.source,
        {
            "run_id": "game-run-001",
            "dataset_kind": "manual-game-route",
            "dataset_split": "blind",
        },
    )
    dataset, _ = _dataset(tmp_path, [poisoned])
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match=r"metadata\.dataset_split"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_rejects_truncated_frame_manifest(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path)
    run_path = dataset / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["frame_count"] = 3
    run_path.write_text(json.dumps(run), encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match="frame_count"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (lambda labels: labels[0].update(split="validation"), "calibration"),
        (lambda labels: labels[0].update(locatable=False), "locatable"),
        (lambda labels: labels[0].update(run_id="other-run"), "run_id"),
        (lambda labels: labels[0].update(sequence=999), "sequence"),
        (lambda labels: labels[0].update(template_id="../escape"), "template_id"),
        (lambda labels: labels[0].update(route_position=[math.inf, 0]), "route_position"),
        (lambda labels: labels[0].update(waypoint_id=""), "waypoint_id"),
    ],
)
def test_calibrate_templates_rejects_invalid_label_contract(
    tmp_path, mutate, error_match: str
) -> None:
    dataset, _ = _dataset(tmp_path)
    labels = [_label()]
    mutate(labels)
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(json.dumps(labels[0], allow_nan=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_rejects_duplicate_template_id(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label(sequence=0), _label(sequence=1)])

    with pytest.raises(ValueError, match=r"template_id.*重复"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_rejects_conflicting_named_roi(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    second_roi = {
        "id": "scene",
        "x": 0.1,
        "y": 0.1,
        "width": 0.5,
        "height": 0.5,
    }
    _write_labels(
        labels_path,
        [_label(sequence=0), _label(sequence=1, template_id="turn", roi=second_roi)],
    )

    with pytest.raises(ValueError, match=r"ROI.*定义不一致"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_rejects_low_texture_crop(tmp_path) -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame = CapturedFrame(
        0,
        1_000,
        image,
        "fixture",
        {
            "run_id": "game-run-001",
            "dataset_kind": "manual-game-route",
            "dataset_split": "calibration",
        },
    )
    dataset, _ = _dataset(tmp_path, [frame])
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match="纹理不足"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_refuses_non_empty_output_directory(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])
    output = tmp_path / "profile"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="输出目录"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=output,
            matcher=MatcherConfiguration.default(),
        )


@pytest.mark.parametrize(
    ("run_update", "error_match"),
    [
        ({"run_id": ""}, "run_id"),
        ({"resolution": [0, 80]}, "resolution"),
        ({"resolution": [True, 80]}, "resolution"),
        ({"resolution": "100x80"}, "resolution"),
    ],
)
def test_calibrate_templates_rejects_invalid_run_contract(
    tmp_path, run_update, error_match
) -> None:
    dataset, _ = _dataset(tmp_path)
    run_path = dataset / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(run_update)
    run_path.write_text(json.dumps(run), encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match=error_match):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


@pytest.mark.parametrize(
    ("metadata_update", "error_match"),
    [
        ({"run_id": "other-run"}, "metadata.run_id"),
        ({"dataset_kind": "synthetic"}, "manual-game-route"),
    ],
)
def test_calibrate_templates_rejects_untrusted_frame_metadata(
    tmp_path, metadata_update, error_match
) -> None:
    dataset, _ = _dataset(tmp_path)
    manifest_path = dataset / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    records[0]["metadata"].update(metadata_update)
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match=error_match):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_rejects_resolution_drift(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path)
    run_path = dataset / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["resolution"] = [101, 80]
    run_path.write_text(json.dumps(run), encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match="分辨率"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_rejects_empty_dataset(tmp_path) -> None:
    dataset, _ = _dataset(tmp_path, [])
    (dataset / "manifest.jsonl").write_text("", encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    with pytest.raises(ValueError, match="frame_count"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=tmp_path / "profile",
            matcher=MatcherConfiguration.default(),
        )


def test_calibrate_templates_validates_all_crops_before_writing(tmp_path) -> None:
    textured = _frame(0, 1)
    plain_image = np.zeros((80, 100, 3), dtype=np.uint8)
    plain_image.setflags(write=False)
    plain = CapturedFrame(
        1,
        1_001,
        plain_image,
        "fixture",
        {
            "run_id": "game-run-001",
            "dataset_kind": "manual-game-route",
            "dataset_split": "calibration",
        },
    )
    dataset, _ = _dataset(tmp_path, [textured, plain])
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(
        labels_path,
        [_label(sequence=0), _label(sequence=1, template_id="plain")],
    )
    output = tmp_path / "profile"

    with pytest.raises(ValueError, match="纹理不足"):
        calibrate_templates(
            dataset_directory=dataset,
            labels_path=labels_path,
            output_directory=output,
            matcher=MatcherConfiguration.default(),
        )

    assert not output.exists()


def test_main_reports_success_and_failure(tmp_path, capsys) -> None:
    dataset, _ = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(labels_path, [_label()])

    assert (
        main(
            [
                "--dataset",
                str(dataset),
                "--labels",
                str(labels_path),
                "--output",
                str(tmp_path / "profile"),
            ]
        )
        == 0
    )
    success = json.loads(capsys.readouterr().out)
    assert success["run_id"] == "game-run-001"
    assert success["template_count"] == 1

    assert (
        main(
            [
                "--dataset",
                str(tmp_path / "missing"),
                "--labels",
                str(labels_path),
                "--output",
                str(tmp_path / "other-profile"),
            ]
        )
        == 1
    )
    assert "模板标定失败" in capsys.readouterr().err


def test_main_can_emit_feature_profile_from_cli_flags(tmp_path, capsys) -> None:
    dataset, _ = _dataset(tmp_path)
    labels_path = tmp_path / "labels.jsonl"
    _write_labels(
        labels_path,
        [
            _label(
                roi={
                    "id": "scene",
                    "x": 0,
                    "y": 0,
                    "width": 1,
                    "height": 1,
                }
            )
        ],
    )
    output = tmp_path / "feature-profile"

    result = main(
        [
            "--dataset",
            str(dataset),
            "--labels",
            str(labels_path),
            "--output",
            str(output),
            "--feature-backend",
            "sift",
            "--maximum-features",
            "1200",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "templates.json").read_text(encoding="utf-8"))

    assert result == 0
    assert payload["feature_backend"] == "sift"
    assert manifest["feature_matcher"]["maximum_features"] == 1200
