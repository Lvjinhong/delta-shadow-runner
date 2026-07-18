"""从人工路线回放和标签生成可追溯的模板 Profile。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureRegion
from .feature_matching import (
    FeatureBackend,
    FeatureMatchPolicy,
    LocalFeatureAnchorDetector,
)
from .frames import DatasetContentDigest, ReplayFrameSource, frame_content_sha256


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _finite_number(value: object, *, field: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f'标签字段 "{field}" 必须是有限数')
    return float(value)


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'标签字段 "{field}" 必须是对象')
    return value


@dataclass(frozen=True, slots=True)
class NormalizedRoi:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("归一化 ROI 坐标必须是有限数")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("归一化 ROI 左上角不能为负，宽高必须为正")
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("归一化 ROI 不能超出 0..1 范围")

    @staticmethod
    def _floor(value: float, size: int) -> int:
        return int((Decimal(str(value)) * size).to_integral_value(rounding=ROUND_FLOOR))

    @staticmethod
    def _ceil(value: Decimal, size: int) -> int:
        return int((value * size).to_integral_value(rounding=ROUND_CEILING))

    def to_region(self, frame_width: int, frame_height: int) -> CaptureRegion:
        if type(frame_width) is not int or frame_width <= 0:
            raise ValueError("frame_width 必须是正整数")
        if type(frame_height) is not int or frame_height <= 0:
            raise ValueError("frame_height 必须是正整数")
        left = self._floor(self.x, frame_width)
        top = self._floor(self.y, frame_height)
        right = self._ceil(Decimal(str(self.x)) + Decimal(str(self.width)), frame_width)
        bottom = self._ceil(Decimal(str(self.y)) + Decimal(str(self.height)), frame_height)
        return CaptureRegion(left, top, right - left, bottom - top)


@dataclass(frozen=True, slots=True)
class MatcherConfiguration:
    scales: tuple[float, ...]
    score_threshold: float
    minimum_spatial_margin: float
    minimum_template_margin: float
    nms_radius_px: int

    def __post_init__(self) -> None:
        if not self.scales or any(
            not _is_finite_number(scale) or scale <= 0 for scale in self.scales
        ):
            raise ValueError("模板缩放比例必须是非空的正有限数")
        if not _is_finite_number(self.score_threshold) or not (0 < self.score_threshold <= 1):
            raise ValueError("模板分数阈值必须位于 0 到 1 之间")
        if not _is_finite_number(self.minimum_spatial_margin) or not (
            0 <= self.minimum_spatial_margin <= 1
        ):
            raise ValueError("空间候选差值必须位于 0 到 1 之间")
        if not _is_finite_number(self.minimum_template_margin) or not (
            0 <= self.minimum_template_margin <= 1
        ):
            raise ValueError("不同 waypoint 的候选差值必须位于 0 到 1 之间")
        if type(self.nms_radius_px) is not int or self.nms_radius_px <= 0:
            raise ValueError("NMS 半径必须是正整数")

    @classmethod
    def default(cls) -> MatcherConfiguration:
        return cls(
            scales=(1.0,),
            score_threshold=0.82,
            minimum_spatial_margin=0.05,
            minimum_template_margin=0.05,
            nms_radius_px=18,
        )


@dataclass(frozen=True, slots=True)
class FeatureMatcherConfiguration:
    backend: FeatureBackend
    policy: FeatureMatchPolicy
    maximum_features: int

    def __post_init__(self) -> None:
        if not isinstance(self.backend, FeatureBackend):
            raise ValueError("特征 backend 必须是 ORB 或 SIFT")
        if not isinstance(self.policy, FeatureMatchPolicy):
            raise ValueError("特征 policy 类型无效")
        if type(self.maximum_features) is not int or not 32 <= self.maximum_features <= 50_000:
            raise ValueError("maximum_features 必须是 32 到 50000 的整数")

    @classmethod
    def default(cls, backend: FeatureBackend) -> FeatureMatcherConfiguration:
        if not isinstance(backend, FeatureBackend):
            raise ValueError("特征 backend 必须是 ORB 或 SIFT")
        return cls(
            backend=backend,
            policy=FeatureMatchPolicy(
                ratio_threshold=0.8 if backend is FeatureBackend.ORB else 0.75,
                ransac_reprojection_threshold_px=3.0,
                minimum_good_matches=12,
                minimum_inliers=10,
                minimum_inlier_ratio=0.55,
                maximum_reprojection_rmse_px=4.0,
                maximum_reprojection_p95_px=6.0,
                minimum_source_coverage=0.05,
                minimum_target_coverage=0.02,
                minimum_projected_area_ratio=0.01,
                maximum_projected_area_ratio=1.0,
                maximum_homography_condition_number=100_000_000.0,
                secondary_minimum_inliers=8,
            ),
            maximum_features=3000,
        )

    def to_manifest(self) -> dict[str, object]:
        policy = self.policy
        return {
            "backend": str(self.backend),
            "maximum_features": self.maximum_features,
            "ratio_threshold": policy.ratio_threshold,
            "ransac_reprojection_threshold_px": policy.ransac_reprojection_threshold_px,
            "minimum_good_matches": policy.minimum_good_matches,
            "minimum_inliers": policy.minimum_inliers,
            "minimum_inlier_ratio": policy.minimum_inlier_ratio,
            "maximum_reprojection_rmse_px": policy.maximum_reprojection_rmse_px,
            "maximum_reprojection_p95_px": policy.maximum_reprojection_p95_px,
            "minimum_source_coverage": policy.minimum_source_coverage,
            "minimum_target_coverage": policy.minimum_target_coverage,
            "minimum_projected_area_ratio": policy.minimum_projected_area_ratio,
            "maximum_projected_area_ratio": policy.maximum_projected_area_ratio,
            "maximum_homography_condition_number": (policy.maximum_homography_condition_number),
            "secondary_minimum_inliers": policy.secondary_minimum_inliers,
            "minimum_projected_edge_px": policy.minimum_projected_edge_px,
            "secondary_maximum_corner_outside_roi_px": (
                policy.secondary_maximum_corner_outside_roi_px
            ),
        }


@dataclass(frozen=True, slots=True)
class CalibrationLabel:
    run_id: str
    sequence: int
    template_id: str
    roi_id: str
    roi: NormalizedRoi
    route_position: tuple[float, float]
    waypoint_id: str | None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    run_id: str
    template_count: int
    frame_size: tuple[int, int]
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class DatasetScan:
    encoded_templates: tuple[tuple[CalibrationLabel, bytes, str], ...]
    frame_hashes: tuple[tuple[int, str], ...]
    perception_sha256s: tuple[str, ...]
    dataset_content_sha256: str


def _parse_label(record: object, *, line_number: int) -> CalibrationLabel:
    item = _mapping(record, field=f"line[{line_number}]")
    run_id = item.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"第 {line_number} 行 run_id 必须是非空字符串")
    sequence = item.get("sequence")
    if type(sequence) is not int or sequence < 0:
        raise ValueError(f"第 {line_number} 行 sequence 必须是非负整数")
    if item.get("split") != "calibration":
        raise ValueError(f"第 {line_number} 行只能来自 calibration split")
    if item.get("locatable") is not True:
        raise ValueError(f"第 {line_number} 行 calibration 标签必须 locatable=true")
    template_id = item.get("template_id")
    if (
        not isinstance(template_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", template_id) is None
    ):
        raise ValueError(f"第 {line_number} 行 template_id 不安全或为空")
    raw_roi = _mapping(item.get("roi"), field=f"line[{line_number}].roi")
    roi_id = raw_roi.get("id")
    if (
        not isinstance(roi_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", roi_id) is None
    ):
        raise ValueError(f"第 {line_number} 行 roi.id 不安全或为空")
    roi = NormalizedRoi(
        raw_roi.get("x"),
        raw_roi.get("y"),
        raw_roi.get("width"),
        raw_roi.get("height"),
    )
    raw_position = item.get("route_position")
    if not isinstance(raw_position, list) or len(raw_position) != 2:
        raise ValueError(f"第 {line_number} 行 route_position 必须包含两个数")
    route_position = (
        _finite_number(raw_position[0], field="route_position[0]"),
        _finite_number(raw_position[1], field="route_position[1]"),
    )
    waypoint_id = item.get("waypoint_id")
    if waypoint_id is not None and (not isinstance(waypoint_id, str) or not waypoint_id):
        raise ValueError(f"第 {line_number} 行 waypoint_id 必须是非空字符串或 null")
    return CalibrationLabel(
        run_id=run_id,
        sequence=sequence,
        template_id=template_id,
        roi_id=roi_id,
        roi=roi,
        route_position=route_position,
        waypoint_id=waypoint_id,
    )


def _load_labels(path: Path) -> tuple[CalibrationLabel, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"标签文件不存在: {path}")
    labels: list[CalibrationLabel] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            labels.append(_parse_label(json.loads(line), line_number=line_number))
    if not labels:
        raise ValueError("标签文件至少需要一条 calibration 标签")
    template_ids = tuple(label.template_id for label in labels)
    if len(set(template_ids)) != len(template_ids):
        raise ValueError("template_id 不能重复")
    rois: dict[str, NormalizedRoi] = {}
    for label in labels:
        previous = rois.setdefault(label.roi_id, label.roi)
        if previous != label.roi:
            raise ValueError(f'同名 ROI "{label.roi_id}" 定义不一致')
    return tuple(labels)


def _load_run(dataset_root: Path) -> tuple[str, tuple[int, int], int, str]:
    run_path = dataset_root / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"采样 run.json 不存在: {run_path}")
    raw_bytes = run_path.read_bytes()
    run = _mapping(json.loads(raw_bytes.decode("utf-8")), field="run.json")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.json 的 run_id 必须是非空字符串")
    dataset_split = run.get("dataset_split")
    if dataset_split != "calibration":
        raise ValueError('run.json 的 dataset_split 必须是 "calibration"')
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


def _scan_dataset(
    dataset_root: Path,
    *,
    run_id: str,
    frame_size: tuple[int, int],
    expected_frame_count: int,
    labels: tuple[CalibrationLabel, ...],
    regions: dict[str, CaptureRegion],
) -> DatasetScan:
    labels_by_sequence: dict[int, list[CalibrationLabel]] = {}
    for label in labels:
        labels_by_sequence.setdefault(label.sequence, []).append(label)

    encoded_templates: list[tuple[CalibrationLabel, bytes, str]] = []
    frame_hashes: list[tuple[int, str]] = []
    perception_sha256s: list[str] = []
    seen_label_sequences: set[int] = set()
    content_digest = DatasetContentDigest()
    replayed_frame_count = 0
    for frame in ReplayFrameSource(dataset_root):
        replayed_frame_count += 1
        actual_size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if actual_size != frame_size:
            raise ValueError(
                "回放帧分辨率与 run.json 不一致: "
                f"expected={frame_size[0]}x{frame_size[1]}, "
                f"actual={actual_size[0]}x{actual_size[1]}"
            )
        if frame.metadata.get("run_id") != run_id:
            raise ValueError(f"sequence={frame.sequence} 的 metadata.run_id 不匹配")
        if frame.metadata.get("dataset_kind") != "manual-game-route":
            raise ValueError(f"sequence={frame.sequence} 不是 manual-game-route 采样数据")
        if frame.metadata.get("dataset_split") != "calibration":
            raise ValueError(
                f"sequence={frame.sequence} 的 metadata.dataset_split 不是 calibration"
            )
        frame_sha256 = content_digest.update(frame.sequence, frame.image)
        frame_hashes.append((frame.sequence, frame_sha256))
        for _roi_id, region in sorted(regions.items()):
            perception_sha256s.append(
                frame_content_sha256(
                    frame.image[
                        region.top : region.bottom,
                        region.left : region.right,
                    ]
                )
            )
        for label in labels_by_sequence.get(frame.sequence, ()):
            region = regions[label.roi_id]
            crop = np.array(
                frame.image[region.top : region.bottom, region.left : region.right],
                copy=True,
            )
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if float(np.std(gray)) < 1:
                raise ValueError(f'模板 "{label.template_id}" 纹理不足')
            encoded, buffer = cv2.imencode(".png", crop)
            if not encoded:
                raise OSError(f'模板 "{label.template_id}" PNG 编码失败')
            encoded_templates.append((label, buffer.tobytes(), frame_sha256))
            seen_label_sequences.add(frame.sequence)

    if replayed_frame_count == 0:
        raise ValueError("采样数据集不包含任何回放帧")
    if replayed_frame_count != expected_frame_count:
        raise ValueError(
            "run.json.frame_count 与回放清单不一致: "
            f"declared={expected_frame_count}, replayed={replayed_frame_count}"
        )
    missing_sequences = sorted(set(labels_by_sequence) - seen_label_sequences)
    if missing_sequences:
        raise ValueError(f"标签 sequence={missing_sequences[0]} 不在采样数据集中")
    return DatasetScan(
        encoded_templates=tuple(encoded_templates),
        frame_hashes=tuple(frame_hashes),
        perception_sha256s=tuple(perception_sha256s),
        dataset_content_sha256=content_digest.hexdigest(),
    )


def calibrate_templates(
    *,
    dataset_directory: str | Path,
    labels_path: str | Path,
    output_directory: str | Path,
    matcher: MatcherConfiguration,
    feature_matcher: FeatureMatcherConfiguration | None = None,
) -> CalibrationResult:
    dataset_root = Path(dataset_directory)
    output_root = Path(output_directory)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"模板输出目录必须为空: {output_root}")
    run_id, frame_size, expected_frame_count, run_json_sha256 = _load_run(dataset_root)
    labels = _load_labels(Path(labels_path))
    if any(label.run_id != run_id for label in labels):
        raise ValueError(f'标签 run_id 必须全部等于采样运行 "{run_id}"')
    frame_width, frame_height = frame_size
    regions: dict[str, CaptureRegion] = {}
    for label in labels:
        region = label.roi.to_region(frame_width, frame_height)
        regions[label.roi_id] = region
    scan = _scan_dataset(
        dataset_root,
        run_id=run_id,
        frame_size=frame_size,
        expected_frame_count=expected_frame_count,
        labels=labels,
        regions=regions,
    )
    if feature_matcher is not None:
        assigned_template_count = 0
        positions_by_waypoint: dict[str, tuple[float, float]] = {}
        for label, image_bytes, _source_frame_sha256 in scan.encoded_templates:
            if label.waypoint_id is None:
                continue
            existing_position = positions_by_waypoint.setdefault(
                label.waypoint_id,
                label.route_position,
            )
            if existing_position != label.route_position:
                raise ValueError("同一 waypoint 的特征路线坐标必须一致")
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError(f'模板 "{label.template_id}" 无法解码')
            LocalFeatureAnchorDetector(
                label=label.template_id,
                waypoint_id=label.waypoint_id,
                template=image,
                search_roi=regions[label.roi_id],
                backend=feature_matcher.backend,
                policy=feature_matcher.policy,
                maximum_features=feature_matcher.maximum_features,
            )
            assigned_template_count += 1
        if assigned_template_count == 0:
            raise ValueError("特征 Profile 至少需要一个有 waypoint_id 的模板")

    templates_root = output_root / "templates"
    templates_root.mkdir(parents=True, exist_ok=True)
    manifest_templates = []
    for label, image_bytes, source_frame_sha256 in scan.encoded_templates:
        relative_image = Path("templates") / f"{label.template_id}.png"
        image_path = output_root / relative_image
        temporary_path = image_path.with_name(f".{image_path.name}.tmp")
        temporary_path.write_bytes(image_bytes)
        temporary_path.replace(image_path)
        manifest_templates.append(
            {
                "id": label.template_id,
                "image": relative_image.as_posix(),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "roi_id": label.roi_id,
                "route_position": list(label.route_position),
                "waypoint_id": label.waypoint_id,
                "source_run_id": label.run_id,
                "source_sequence": label.sequence,
                "source_frame_sha256": source_frame_sha256,
            }
        )
    manifest = {
        "schema_version": 2,
        "capture_profile": {"width": frame_width, "height": frame_height},
        "matcher": {
            "scales": list(matcher.scales),
            "score_threshold": matcher.score_threshold,
            "minimum_spatial_margin": matcher.minimum_spatial_margin,
            "minimum_template_margin": matcher.minimum_template_margin,
            "nms_radius_px": matcher.nms_radius_px,
        },
        "rois": {
            roi_id: {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }
            for roi_id, region in sorted(regions.items())
        },
        "source_datasets": [
            {
                "run_id": run_id,
                "frame_sha256s": [sha256 for _sequence, sha256 in scan.frame_hashes],
                "frame_hashes": [
                    {"sequence": sequence, "sha256": sha256}
                    for sequence, sha256 in scan.frame_hashes
                ],
                "perception_sha256s": list(scan.perception_sha256s),
                "dataset_content_sha256": scan.dataset_content_sha256,
                "run_json_sha256": run_json_sha256,
                "frame_manifest_sha256": hashlib.sha256(
                    (dataset_root / "manifest.jsonl").read_bytes()
                ).hexdigest(),
            }
        ],
        "templates": manifest_templates,
    }
    if feature_matcher is not None:
        manifest["feature_matcher"] = feature_matcher.to_manifest()
    manifest_path = output_root / "templates.json"
    temporary_manifest = output_root / ".templates.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return CalibrationResult(
        run_id=run_id,
        template_count=len(manifest_templates),
        frame_size=frame_size,
        manifest_path=manifest_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从人工路线截图和标签生成模板 Profile")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-backend",
        choices=("ncc", "orb", "sift"),
        default="ncc",
    )
    parser.add_argument("--maximum-features", type=int, default=3000)
    args = parser.parse_args(argv)
    try:
        feature_matcher = None
        if args.feature_backend != "ncc":
            default_feature_matcher = FeatureMatcherConfiguration.default(
                FeatureBackend(args.feature_backend)
            )
            feature_matcher = FeatureMatcherConfiguration(
                backend=default_feature_matcher.backend,
                policy=default_feature_matcher.policy,
                maximum_features=args.maximum_features,
            )
        result = calibrate_templates(
            dataset_directory=args.dataset,
            labels_path=args.labels,
            output_directory=args.output,
            matcher=MatcherConfiguration.default(),
            feature_matcher=feature_matcher,
        )
    except Exception as error:
        print(f"模板标定失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "template_count": result.template_count,
                "frame_size": result.frame_size,
                "manifest_path": str(result.manifest_path),
                "feature_backend": args.feature_backend,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
