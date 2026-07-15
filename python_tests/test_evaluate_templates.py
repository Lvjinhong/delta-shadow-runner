import json
import math

import numpy as np
import pytest

from delta_vision.calibrate_templates import MatcherConfiguration, calibrate_templates
from delta_vision.evaluate_templates import evaluate_template_profile, main
from delta_vision.frames import CapturedFrame, FrameRecorder

FRAME_WIDTH = 100
FRAME_HEIGHT = 80
ROI = {
    "id": "scene",
    "x": 0.1,
    "y": 0.1,
    "width": 0.8,
    "height": 0.8,
}


def _image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        256,
        size=(FRAME_HEIGHT, FRAME_WIDTH, 3),
        dtype=np.uint8,
    )


def _frame(sequence: int, image: np.ndarray, *, run_id: str) -> CapturedFrame:
    owned = np.array(image, copy=True)
    owned.setflags(write=False)
    return CapturedFrame(
        sequence,
        1_000 + sequence,
        owned,
        "fixture",
        {
            "run_id": run_id,
            "dataset_kind": "manual-game-route",
        },
    )


def _record_dataset(
    root,
    *,
    run_id: str,
    images: list[np.ndarray],
    dataset_split: str = "blind",
):
    recorder = FrameRecorder(root)
    for sequence, image in enumerate(images):
        recorder.record(
            _frame(sequence, image, run_id=run_id),
            metadata={
                "run_id": run_id,
                "dataset_kind": "manual-game-route",
                "dataset_split": dataset_split,
            },
        )
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "window_title": "三角洲行动",
                "backend": "fixture",
                "dataset_split": dataset_split,
                "frame_count": len(images),
                "resolution": [FRAME_WIDTH, FRAME_HEIGHT],
            }
        ),
        encoding="utf-8",
    )
    return root


def _same_roi_different_frame(image: np.ndarray, value: int) -> np.ndarray:
    """保留模板 ROI，只修改 ROI 外围，避免 blind 集复用标定原帧。"""
    changed = np.array(image, copy=True)
    changed[:8, :] = value
    return changed


def _matching_variant(image: np.ndarray, value: int) -> np.ndarray:
    """构造不复用标定 ROI 字节、但仍应被同一模板识别的回放帧。"""
    changed = _same_roi_different_frame(image, value)
    roi = changed[8:72, 10:90].astype(np.int16)
    changed[8:72, 10:90] = np.clip(roi + value % 5 + 1, 0, 255).astype(np.uint8)
    return changed


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _calibration_label(sequence: int, template_id: str, position) -> dict:
    return {
        "run_id": "calibration-run",
        "sequence": sequence,
        "split": "calibration",
        "locatable": True,
        "template_id": template_id,
        "roi": ROI,
        "route_position": list(position),
        "waypoint_id": template_id,
    }


def _evaluation_label(
    sequence: int,
    *,
    locatable: bool,
    position=None,
    split: str = "blind",
    run_id: str = "blind-run",
    expected_waypoint_id: str | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "sequence": sequence,
        "split": split,
        "locatable": locatable,
        "route_position": list(position) if position is not None else None,
        "expected_waypoint_id": ((expected_waypoint_id or "start") if locatable else None),
    }


def _profile(tmp_path):
    first = _image(1)
    second = _image(2)
    calibration = _record_dataset(
        tmp_path / "calibration",
        run_id="calibration-run",
        images=[first, second],
        dataset_split="calibration",
    )
    labels_path = tmp_path / "calibration-labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            _calibration_label(0, "start", (0, 0)),
            _calibration_label(1, "turn", (100, 0)),
        ],
    )
    result = calibrate_templates(
        dataset_directory=calibration,
        labels_path=labels_path,
        output_directory=tmp_path / "profile",
        matcher=MatcherConfiguration.default(),
    )
    return result.manifest_path, first, second


def test_evaluate_template_profile_reports_blind_metrics_and_predictions(
    tmp_path,
) -> None:
    profile_path, first, second = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[
            _matching_variant(first, 11),
            _matching_variant(second, 22),
            np.zeros_like(first),
        ],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            _evaluation_label(
                0,
                locatable=True,
                position=(0, 0),
                expected_waypoint_id="start",
            ),
            _evaluation_label(
                1,
                locatable=True,
                position=(100, 0),
                expected_waypoint_id="turn",
            ),
            _evaluation_label(2, locatable=False),
        ],
    )

    result = evaluate_template_profile(
        profile_path=profile_path,
        dataset_directory=blind,
        labels_path=labels_path,
        output_directory=tmp_path / "evaluation",
        split="blind",
        distance_tolerance=25,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert result.sample_count == 3
    assert metrics["dataset_run_id"] == "blind-run"
    assert len(metrics["dataset_run_json_sha256"]) == 64
    assert len(metrics["dataset_frame_manifest_sha256"]) == 64
    assert len(metrics["dataset_content_sha256"]) == 64
    assert metrics["split"] == "blind"
    assert metrics["sample_count"] == 3
    assert metrics["locatable_count"] == 2
    assert metrics["unlocatable_count"] == 1
    assert metrics["waypoint_top1_accuracy"] == 1
    assert metrics["exact_position_match_rate"] == 1
    assert metrics["within_distance_tolerance_rate"] == 1
    assert metrics["pose_availability"] == 1
    assert metrics["pose_emission_precision"] == 1
    assert metrics["pose_emission_recall"] == 1
    assert metrics["pose_emission_f1"] == 1
    assert metrics["balanced_pose_emission_accuracy"] == 1
    assert metrics["false_lock_rate"] == 0
    assert metrics["dataset_frame_count"] == 3
    assert metrics["labeled_frame_count"] == 3
    assert metrics["label_coverage_rate"] == 1
    assert metrics["missed_pose_count"] == 0
    assert metrics["position_error_on_available_poses_route_units"] == {
        "count": 2,
        "median": 0,
        "p90": 0,
        "p95": 0,
        "max": 0,
    }
    predictions = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["sequence"] for item in predictions] == [0, 1, 2]
    assert predictions[0]["predicted_position"] == [0.0, 0.0]
    assert predictions[2]["predicted_position"] is None
    assert predictions[2]["correct_rejection"] is True


def test_evaluate_template_profile_separates_exact_and_tolerance_accuracy(
    tmp_path,
) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(3, 4))],
    )

    result = evaluate_template_profile(
        profile_path=profile_path,
        dataset_directory=blind,
        labels_path=labels_path,
        output_directory=tmp_path / "evaluation",
        split="blind",
        distance_tolerance=5,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["waypoint_top1_accuracy"] == 1
    assert metrics["exact_position_match_rate"] == 0
    assert metrics["within_distance_tolerance_rate"] == 1
    assert metrics["position_error_on_available_poses_route_units"]["median"] == 5


def test_evaluate_template_profile_counts_false_lock(tmp_path) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(labels_path, [_evaluation_label(0, locatable=False)])

    result = evaluate_template_profile(
        profile_path=profile_path,
        dataset_directory=blind,
        labels_path=labels_path,
        output_directory=tmp_path / "evaluation",
        split="blind",
        distance_tolerance=25,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["false_lock_count"] == 1
    assert metrics["false_lock_rate"] == 1
    assert metrics["pose_availability"] is None
    assert metrics["pose_emission_precision"] == 0
    assert metrics["pose_emission_f1"] == 0


def test_evaluate_template_profile_compares_waypoint_id_not_only_position(
    tmp_path,
) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            _evaluation_label(
                0,
                locatable=True,
                position=(0, 0),
                expected_waypoint_id="turn",
            )
        ],
    )

    result = evaluate_template_profile(
        profile_path=profile_path,
        dataset_directory=blind,
        labels_path=labels_path,
        output_directory=tmp_path / "evaluation",
        split="blind",
        distance_tolerance=25,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["waypoint_top1_accuracy"] == 0
    assert metrics["exact_position_match_rate"] == 1


def test_evaluate_template_profile_reports_unavailable_pose_without_nan(
    tmp_path,
) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[np.zeros_like(first)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )

    result = evaluate_template_profile(
        profile_path=profile_path,
        dataset_directory=blind,
        labels_path=labels_path,
        output_directory=tmp_path / "evaluation",
        split="blind",
        distance_tolerance=25,
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["pose_availability"] == 0
    assert metrics["pose_emission_precision"] is None
    assert metrics["pose_emission_recall"] == 0
    assert metrics["pose_emission_f1"] == 0
    assert metrics["waypoint_top1_accuracy"] == 0
    assert metrics["exact_position_match_rate"] == 0
    assert metrics["false_lock_rate"] is None
    assert metrics["position_error_on_available_poses_route_units"] == {
        "count": 0,
        "median": None,
        "p90": None,
        "p95": None,
        "max": None,
    }


def test_evaluate_template_profile_rejects_calibration_run_leakage(tmp_path) -> None:
    profile_path, first, _ = _profile(tmp_path)
    leaked = _record_dataset(
        tmp_path / "leaked",
        run_id="calibration-run",
        images=[first],
    )
    labels_path = tmp_path / "leaked-labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            _evaluation_label(
                0,
                locatable=True,
                position=(0, 0),
                run_id="calibration-run",
            )
        ],
    )

    with pytest.raises(ValueError, match="数据泄漏"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=leaked,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_rejects_same_frame_under_new_run_id(
    tmp_path,
) -> None:
    profile_path, first, _ = _profile(tmp_path)
    leaked = _record_dataset(
        tmp_path / "leaked",
        run_id="renamed-blind-run",
        images=[first],
    )
    labels_path = tmp_path / "leaked-labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            _evaluation_label(
                0,
                locatable=True,
                position=(0, 0),
                run_id="renamed-blind-run",
            )
        ],
    )

    with pytest.raises(ValueError, match=r"帧内容.*泄漏"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=leaked,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_rejects_reused_perception_roi(
    tmp_path,
) -> None:
    profile_path, first, _ = _profile(tmp_path)
    leaked = _record_dataset(
        tmp_path / "leaked",
        run_id="renamed-blind-run",
        images=[_same_roi_different_frame(first, 11)],
    )
    labels_path = tmp_path / "leaked-labels.jsonl"
    _write_jsonl(
        labels_path,
        [
            _evaluation_label(
                0,
                locatable=True,
                position=(0, 0),
                run_id="renamed-blind-run",
            )
        ],
    )

    with pytest.raises(ValueError, match=r"感知 ROI.*泄漏"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=leaked,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (lambda labels: labels[0].update(split="validation"), "split"),
        (lambda labels: labels[0].update(run_id="other-run"), "run_id"),
        (lambda labels: labels[0].update(sequence=-1), "sequence"),
        (lambda labels: labels[0].update(locatable="yes"), "locatable"),
        (lambda labels: labels[0].update(expected_waypoint_id=None), "waypoint"),
        (lambda labels: labels[0].update(route_position=None), "route_position"),
        (
            lambda labels: labels[0].update(route_position=[math.inf, 0]),
            "route_position",
        ),
    ],
)
def test_evaluate_template_profile_rejects_invalid_label_contract(
    tmp_path, mutate, error_match
) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels = [_evaluation_label(0, locatable=True, position=(0, 0))]
    mutate(labels)
    labels_path = tmp_path / "blind-labels.jsonl"
    labels_path.write_text(
        json.dumps(labels[0], allow_nan=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error_match):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_rejects_duplicate_sequence(tmp_path) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    label = _evaluation_label(0, locatable=True, position=(0, 0))
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(labels_path, [label, label])

    with pytest.raises(ValueError, match=r"sequence.*重复"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_requires_labeled_sequence_in_dataset(
    tmp_path,
) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(99, locatable=True, position=(0, 0))],
    )

    with pytest.raises(ValueError, match="sequence=99"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_requires_labels_for_every_dataset_frame(
    tmp_path,
) -> None:
    profile_path, first, second = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[
            _matching_variant(first, 11),
            _matching_variant(second, 22),
        ],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )

    with pytest.raises(ValueError, match=r"未标注.*sequence=1"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_rejects_truncated_frame_manifest(tmp_path) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    run_path = blind / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["frame_count"] = 2
    run_path.write_text(json.dumps(run), encoding="utf-8")
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )

    with pytest.raises(ValueError, match="frame_count"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_requires_dataset_declared_split(tmp_path) -> None:
    profile_path, first, _ = _profile(tmp_path)
    validation = _record_dataset(
        tmp_path / "validation",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
        dataset_split="validation",
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )

    with pytest.raises(ValueError, match="dataset_split"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=validation,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


def test_evaluate_template_profile_rejects_resolution_mismatch(tmp_path) -> None:
    profile_path, _first, _ = _profile(tmp_path)
    larger = np.zeros((FRAME_HEIGHT, FRAME_WIDTH + 1, 3), dtype=np.uint8)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[larger],
    )
    run_path = blind / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["resolution"] = [FRAME_WIDTH + 1, FRAME_HEIGHT]
    run_path.write_text(json.dumps(run), encoding="utf-8")
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )

    with pytest.raises(ValueError, match=r"Profile.*分辨率"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=25,
        )


@pytest.mark.parametrize("distance_tolerance", [0, -1, math.nan, True, "25"])
def test_evaluate_template_profile_rejects_invalid_tolerance(tmp_path, distance_tolerance) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )

    with pytest.raises(ValueError, match="tolerance"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=tmp_path / "evaluation",
            split="blind",
            distance_tolerance=distance_tolerance,
        )


def test_evaluate_template_profile_refuses_non_empty_output(tmp_path) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="输出目录"):
        evaluate_template_profile(
            profile_path=profile_path,
            dataset_directory=blind,
            labels_path=labels_path,
            output_directory=output,
            split="blind",
            distance_tolerance=25,
        )


def test_main_reports_success_and_failure(tmp_path, capsys) -> None:
    profile_path, first, _ = _profile(tmp_path)
    blind = _record_dataset(
        tmp_path / "blind",
        run_id="blind-run",
        images=[_matching_variant(first, 11)],
    )
    labels_path = tmp_path / "blind-labels.jsonl"
    _write_jsonl(
        labels_path,
        [_evaluation_label(0, locatable=True, position=(0, 0))],
    )

    assert (
        main(
            [
                "--profile",
                str(profile_path),
                "--dataset",
                str(blind),
                "--labels",
                str(labels_path),
                "--output",
                str(tmp_path / "evaluation"),
                "--split",
                "blind",
                "--distance-tolerance",
                "25",
            ]
        )
        == 0
    )
    success = json.loads(capsys.readouterr().out)
    assert success["sample_count"] == 1

    assert (
        main(
            [
                "--profile",
                str(profile_path),
                "--dataset",
                str(tmp_path / "missing"),
                "--labels",
                str(labels_path),
                "--output",
                str(tmp_path / "other-evaluation"),
                "--split",
                "blind",
            ]
        )
        == 1
    )
    assert "模板评估失败" in capsys.readouterr().err
