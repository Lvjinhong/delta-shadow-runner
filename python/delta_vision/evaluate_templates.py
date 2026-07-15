"""在独立运行的完整路线数据集上评估模板 Profile。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import DatasetContentDigest, ReplayFrameSource, frame_content_sha256
from .template_profile import load_template_profile


@dataclass(frozen=True, slots=True)
class EvaluationLabel:
    run_id: str
    sequence: int
    split: str
    locatable: bool
    route_position: tuple[float, float] | None
    expected_waypoint_id: str | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    dataset_run_id: str
    split: str
    sample_count: int
    metrics_path: Path
    predictions_path: Path


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'评估字段 "{field}" 必须是对象')
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'评估字段 "{field}" 必须是有限数')
    return float(value)


def _parse_label(
    record: object,
    *,
    line_number: int,
    expected_split: str,
) -> EvaluationLabel:
    item = _mapping(record, field=f"line[{line_number}]")
    run_id = item.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"第 {line_number} 行 run_id 必须是非空字符串")
    sequence = item.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError(f"第 {line_number} 行 sequence 必须是非负整数")
    split = item.get("split")
    if split != expected_split:
        raise ValueError(f'第 {line_number} 行 split 必须是 "{expected_split}"，实际为 {split!r}')
    locatable = item.get("locatable")
    if type(locatable) is not bool:
        raise ValueError(f"第 {line_number} 行 locatable 必须是布尔值")
    raw_position = item.get("route_position")
    waypoint_id = item.get("expected_waypoint_id")
    if locatable:
        if not isinstance(raw_position, list) or len(raw_position) != 2:
            raise ValueError(f"第 {line_number} 行可定位样本的 route_position 必须包含两个数")
        route_position = (
            _finite_number(raw_position[0], field="route_position[0]"),
            _finite_number(raw_position[1], field="route_position[1]"),
        )
        if not isinstance(waypoint_id, str) or not waypoint_id:
            raise ValueError(f"第 {line_number} 行可定位样本的 expected_waypoint_id 必须非空")
    else:
        if raw_position is not None:
            raise ValueError(f"第 {line_number} 行不可定位样本的 route_position 必须是 null")
        if waypoint_id is not None:
            raise ValueError(f"第 {line_number} 行不可定位样本的 expected_waypoint_id 必须是 null")
        route_position = None
    return EvaluationLabel(
        run_id=run_id,
        sequence=sequence,
        split=split,
        locatable=locatable,
        route_position=route_position,
        expected_waypoint_id=waypoint_id,
    )


def _load_labels(
    path: Path,
    *,
    split: str,
) -> tuple[tuple[EvaluationLabel, ...], str]:
    if not path.is_file():
        raise FileNotFoundError(f"评估标签文件不存在: {path}")
    raw_bytes = path.read_bytes()
    labels: list[EvaluationLabel] = []
    for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"无法解析 {path} 第 {line_number} 行: {error.msg}") from error
        labels.append(_parse_label(record, line_number=line_number, expected_split=split))
    if not labels:
        raise ValueError("评估标签文件至少需要一条标签")
    sequences = tuple(label.sequence for label in labels)
    if len(set(sequences)) != len(sequences):
        raise ValueError("评估标签 sequence 不能重复")
    return (
        tuple(sorted(labels, key=lambda label: label.sequence)),
        hashlib.sha256(raw_bytes).hexdigest(),
    )


def _load_run(
    dataset_root: Path,
    *,
    expected_split: str,
) -> tuple[str, tuple[int, int], int, str]:
    run_path = dataset_root / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"评估 run.json 不存在: {run_path}")
    raw_bytes = run_path.read_bytes()
    run = _mapping(json.loads(raw_bytes.decode("utf-8")), field="run.json")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.json 的 run_id 必须是非空字符串")
    dataset_split = run.get("dataset_split")
    if dataset_split != expected_split:
        raise ValueError(
            f'run.json 的 dataset_split 必须是 "{expected_split}"，实际为 {dataset_split!r}'
        )
    frame_count = run.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("run.json 的 frame_count 必须是正整数")
    resolution = run.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(value) is not int or value <= 0 for value in resolution)
    ):
        raise ValueError("run.json 的 resolution 必须包含两个正整数")
    return (
        run_id,
        (resolution[0], resolution[1]),
        frame_count,
        hashlib.sha256(raw_bytes).hexdigest(),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1_counts(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else None


def _position_summary(errors: list[float]) -> dict[str, float | int | None]:
    if not errors:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}
    percentiles = np.percentile(np.asarray(errors, dtype=np.float64), [50, 90, 95])
    return {
        "count": len(errors),
        "median": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "max": float(max(errors)),
    }


def evaluate_template_profile(
    *,
    profile_path: str | Path,
    dataset_directory: str | Path,
    labels_path: str | Path,
    output_directory: str | Path,
    split: str,
    distance_tolerance: float,
) -> EvaluationResult:
    if split not in {"validation", "blind"}:
        raise ValueError('split 必须是 "validation" 或 "blind"')
    if (
        isinstance(distance_tolerance, bool)
        or not isinstance(distance_tolerance, (int, float))
        or not math.isfinite(distance_tolerance)
        or distance_tolerance <= 0
    ):
        raise ValueError("distance_tolerance 必须是正有限数")
    output_root = Path(output_directory)
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"评估输出目录必须为空: {output_root}")

    profile = load_template_profile(profile_path)
    dataset_root = Path(dataset_directory)
    run_id, frame_size, declared_frame_count, run_json_sha256 = _load_run(
        dataset_root,
        expected_split=split,
    )
    frame_manifest_path = dataset_root / "manifest.jsonl"
    if not frame_manifest_path.is_file():
        raise FileNotFoundError(f"评估帧清单不存在: {frame_manifest_path}")
    frame_manifest_sha256 = hashlib.sha256(frame_manifest_path.read_bytes()).hexdigest()
    if run_id in profile.source_run_ids:
        raise ValueError(f'评估运行 "{run_id}" 已用于模板标定，存在数据泄漏')
    if frame_size != profile.frame_size:
        raise ValueError(
            "Profile 与评估数据分辨率不一致: "
            f"profile={profile.frame_size[0]}x{profile.frame_size[1]}, "
            f"dataset={frame_size[0]}x{frame_size[1]}"
        )
    labels, labels_sha256 = _load_labels(Path(labels_path), split=split)
    if any(label.run_id != run_id for label in labels):
        raise ValueError(f'标签 run_id 必须全部等于评估运行 "{run_id}"')
    labels_by_sequence = {label.sequence: label for label in labels}

    predictions: list[dict[str, object]] = []
    seen_sequences: set[int] = set()
    unlabeled_sequences: list[int] = []
    replayed_frame_count = 0
    locatable_count = unlocatable_count = available_count = 0
    waypoint_correct_count = exact_position_count = within_tolerance_count = 0
    false_lock_count = correct_rejection_count = 0
    position_errors: list[float] = []
    dataset_content_digest = DatasetContentDigest()

    for frame in ReplayFrameSource(dataset_root):
        replayed_frame_count += 1
        actual_size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if actual_size != frame_size:
            raise ValueError(
                f"sequence={frame.sequence} 分辨率与 run.json 不一致: "
                f"expected={frame_size}, actual={actual_size}"
            )
        if frame.metadata.get("run_id") != run_id:
            raise ValueError(f"sequence={frame.sequence} 的 metadata.run_id 不匹配")
        if frame.metadata.get("dataset_kind") != "manual-game-route":
            raise ValueError(f"sequence={frame.sequence} 不是 manual-game-route 采样数据")
        if frame.metadata.get("dataset_split") != split:
            raise ValueError(f"sequence={frame.sequence} 的 metadata.dataset_split 不匹配")
        frame_sha256 = dataset_content_digest.update(frame.sequence, frame.image)
        if frame_sha256 in profile.source_frame_sha256s:
            raise ValueError(f"sequence={frame.sequence} 帧内容已出现在标定数据中，存在数据泄漏")
        for region in profile.perception_regions:
            perception_image = frame.image[
                region.top : region.bottom,
                region.left : region.right,
            ]
            if frame_content_sha256(perception_image) in profile.source_perception_sha256s:
                raise ValueError(
                    f"sequence={frame.sequence} 感知 ROI 已出现在标定数据中，存在数据泄漏"
                )
        label = labels_by_sequence.get(frame.sequence)
        if label is None:
            unlabeled_sequences.append(frame.sequence)
            continue
        seen_sequences.add(frame.sequence)
        observation = profile.observer.observe(frame)
        predicted_position = observation.centroid
        error = None
        exact_position = None
        within_tolerance = None
        waypoint_correct = None
        correct_rejection = None

        if label.locatable:
            locatable_count += 1
            if predicted_position is not None:
                available_count += 1
                if label.route_position is None:
                    raise RuntimeError("可定位标签缺少路线坐标")
                error = math.dist(predicted_position, label.route_position)
                if not math.isfinite(error):
                    raise ValueError(f"sequence={frame.sequence} 位置误差溢出")
                position_errors.append(error)
                exact_position = error <= 1e-9
                within_tolerance = error <= distance_tolerance
                waypoint_correct = observation.waypoint_id == label.expected_waypoint_id
                exact_position_count += int(exact_position)
                within_tolerance_count += int(within_tolerance)
                waypoint_correct_count += int(waypoint_correct)
        else:
            unlocatable_count += 1
            false_lock = predicted_position is not None
            false_lock_count += int(false_lock)
            correct_rejection = not false_lock
            correct_rejection_count += int(correct_rejection)

        predictions.append(
            {
                "run_id": run_id,
                "sequence": label.sequence,
                "split": split,
                "locatable": label.locatable,
                "expected_waypoint_id": label.expected_waypoint_id,
                "predicted_waypoint_id": observation.waypoint_id,
                "expected_position": list(label.route_position) if label.route_position else None,
                "predicted_position": list(predicted_position) if predicted_position else None,
                "confidence": observation.confidence,
                "position_error_route_units": error,
                "exact_position_match": exact_position,
                "within_distance_tolerance": within_tolerance,
                "waypoint_top1_correct": waypoint_correct,
                "correct_rejection": correct_rejection,
            }
        )

    if replayed_frame_count != declared_frame_count:
        raise ValueError(
            "run.json.frame_count 与回放清单不一致: "
            f"declared={declared_frame_count}, replayed={replayed_frame_count}"
        )
    missing_sequences = sorted(set(labels_by_sequence) - seen_sequences)
    if unlabeled_sequences or missing_sequences:
        problems = []
        if unlabeled_sequences:
            problems.append(f"数据集未标注 sequence={unlabeled_sequences[0]}")
        if missing_sequences:
            problems.append(f"标签 sequence={missing_sequences[0]} 不在数据集中")
        raise ValueError("评估标签必须完整覆盖数据集: " + "；".join(problems))
    if not seen_sequences:
        raise ValueError("评估数据集不包含任何回放帧")

    missed_pose_count = locatable_count - available_count
    pose_precision = _rate(available_count, available_count + false_lock_count)
    pose_recall = _rate(available_count, locatable_count)
    rejection_rate = _rate(correct_rejection_count, unlocatable_count)
    balanced_pose_accuracy = (
        (pose_recall + rejection_rate) / 2
        if pose_recall is not None and rejection_rate is not None
        else None
    )
    metrics = {
        "schema_version": 1,
        "profile_manifest_sha256": profile.manifest_sha256,
        "profile_source_run_ids": sorted(profile.source_run_ids),
        "dataset_run_id": run_id,
        "dataset_run_json_sha256": run_json_sha256,
        "dataset_frame_manifest_sha256": frame_manifest_sha256,
        "dataset_content_sha256": dataset_content_digest.hexdigest(),
        "labels_sha256": labels_sha256,
        "split": split,
        "frame_size": list(frame_size),
        "distance_tolerance_route_units": float(distance_tolerance),
        "dataset_frame_count": len(seen_sequences),
        "labeled_frame_count": len(labels),
        "label_coverage_rate": 1.0,
        "sample_count": len(labels),
        "locatable_count": locatable_count,
        "unlocatable_count": unlocatable_count,
        "available_pose_count": available_count,
        "missed_pose_count": missed_pose_count,
        "waypoint_top1_correct_count": waypoint_correct_count,
        "exact_position_match_count": exact_position_count,
        "within_distance_tolerance_count": within_tolerance_count,
        "false_lock_count": false_lock_count,
        "correct_rejection_count": correct_rejection_count,
        "waypoint_top1_accuracy": _rate(waypoint_correct_count, locatable_count),
        "exact_position_match_rate": _rate(exact_position_count, locatable_count),
        "within_distance_tolerance_rate": _rate(within_tolerance_count, locatable_count),
        "pose_availability": pose_recall,
        "pose_emission_precision": pose_precision,
        "pose_emission_recall": pose_recall,
        "pose_emission_f1": _f1_counts(
            available_count,
            false_lock_count,
            missed_pose_count,
        ),
        "balanced_pose_emission_accuracy": balanced_pose_accuracy,
        "false_lock_rate": _rate(false_lock_count, unlocatable_count),
        "exact_or_correct_rejection_micro_accuracy": _rate(
            exact_position_count + correct_rejection_count, len(labels)
        ),
        "position_error_on_available_poses_route_units": _position_summary(position_errors),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    temporary_predictions = output_root / ".predictions.jsonl.tmp"
    temporary_predictions.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for item in predictions
        ),
        encoding="utf-8",
    )
    temporary_predictions.replace(predictions_path)
    metrics_path = output_root / "metrics.json"
    temporary_metrics = output_root / ".metrics.json.tmp"
    temporary_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary_metrics.replace(metrics_path)
    return EvaluationResult(
        dataset_run_id=run_id,
        split=split,
        sample_count=len(labels),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在独立路线截图上评估模板准确率")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "blind"), required=True)
    parser.add_argument("--distance-tolerance", type=float, default=25.0)
    args = parser.parse_args(argv)
    try:
        result = evaluate_template_profile(
            profile_path=args.profile,
            dataset_directory=args.dataset,
            labels_path=args.labels,
            output_directory=args.output,
            split=args.split,
            distance_tolerance=args.distance_tolerance,
        )
    except Exception as error:
        print(f"模板评估失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "dataset_run_id": result.dataset_run_id,
                "split": result.split,
                "sample_count": result.sample_count,
                "metrics_path": str(result.metrics_path),
                "predictions_path": str(result.predictions_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
