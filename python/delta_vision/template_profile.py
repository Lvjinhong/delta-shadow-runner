"""可追溯、固定分辨率的路线模板清单加载器。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import numpy as np

from .config import CaptureRegion
from .frames import DatasetContentDigest
from .template_matching import (
    MatchDecisionPolicy,
    RouteTemplate,
    TemplateAnchorDetector,
    TemplateWaypointObserver,
)


@dataclass(frozen=True, slots=True)
class TemplateProfile:
    observer: TemplateWaypointObserver
    frame_size: tuple[int, int]
    manifest_sha256: str
    source_run_ids: frozenset[str]
    source_frame_sha256s: frozenset[str]
    source_perception_sha256s: frozenset[str]
    perception_regions: tuple[CaptureRegion, ...]


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f'模板清单字段 "{field}" 必须是对象')
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'模板清单字段 "{field}" 必须是正整数')
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'模板清单字段 "{field}" 必须是有限数')
    return float(value)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f'模板清单字段 "{field}" 必须是 64 位十六进制')
    return value.lower()


def _safe_asset_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f'模板清单字段 "{field}" 必须是非空相对路径')
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
        or any(":" in part for part in windows_path.parts)
    ):
        raise ValueError(f'模板图像必须位于模板目录内: "{value}"')
    root = root.resolve()
    # 清单统一按相对路径解释，同时兼容 Windows 反斜杠分隔符。
    candidate = (root / value.replace("\\", "/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f'模板图像必须位于模板目录内: "{value}"') from error
    if not candidate.is_file():
        raise FileNotFoundError(f"模板图像不存在: {candidate}")
    return candidate


def _parse_rois(
    raw_rois: object,
    *,
    frame_size: tuple[int, int],
) -> dict[str, CaptureRegion]:
    rois = _mapping(raw_rois, field="rois")
    parsed: dict[str, CaptureRegion] = {}
    frame_width, frame_height = frame_size
    for roi_id, raw_roi in rois.items():
        if not roi_id:
            raise ValueError("ROI ID 不能为空")
        roi = _mapping(raw_roi, field=f"rois.{roi_id}")
        values = {field: roi.get(field) for field in ("left", "top", "width", "height")}
        if any(type(value) is not int for value in values.values()):
            raise ValueError(f'ROI "{roi_id}" 坐标和宽高必须是整数')
        if values["left"] < 0 or values["top"] < 0:
            raise ValueError(f'ROI "{roi_id}" 左上角不能为负数')
        region = CaptureRegion(
            left=values["left"],
            top=values["top"],
            width=values["width"],
            height=values["height"],
        )
        if region.right > frame_width or region.bottom > frame_height:
            raise ValueError(f'ROI "{roi_id}" 不能超出 capture_profile')
        parsed[roi_id] = region
    if not parsed:
        raise ValueError("至少需要一个 ROI")
    return parsed


def load_template_profile(path: str | Path) -> TemplateProfile:
    manifest_path = Path(path)
    raw_bytes = manifest_path.read_bytes()
    raw = _mapping(json.loads(raw_bytes.decode("utf-8")), field="root")
    schema_version = raw.get("schema_version")
    if schema_version == 1 and type(schema_version) is int:
        raise ValueError(
            "template profile schema_version=1 已过期，请重新标定生成 schema_version=2"
        )
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("只支持 template profile schema_version=2")

    capture = _mapping(raw.get("capture_profile"), field="capture_profile")
    frame_size = (
        _positive_int(capture.get("width"), field="capture_profile.width"),
        _positive_int(capture.get("height"), field="capture_profile.height"),
    )
    rois = _parse_rois(raw.get("rois"), frame_size=frame_size)

    matcher = _mapping(raw.get("matcher"), field="matcher")
    raw_scales = matcher.get("scales")
    if not isinstance(raw_scales, list) or not raw_scales:
        raise ValueError('模板清单字段 "matcher.scales" 必须是非空数组')
    scales = tuple(_finite_number(value, field="matcher.scales") for value in raw_scales)
    if any(scale <= 0 for scale in scales):
        raise ValueError('模板清单字段 "matcher.scales" 必须全为正数')
    spatial_policy = MatchDecisionPolicy(
        score_threshold=_finite_number(
            matcher.get("score_threshold"), field="matcher.score_threshold"
        ),
        minimum_margin=_finite_number(
            matcher.get("minimum_spatial_margin"),
            field="matcher.minimum_spatial_margin",
        ),
    )
    template_margin = _finite_number(
        matcher.get("minimum_template_margin"),
        field="matcher.minimum_template_margin",
    )
    nms_radius_px = _positive_int(matcher.get("nms_radius_px"), field="matcher.nms_radius_px")

    raw_source_datasets = raw.get("source_datasets")
    if not isinstance(raw_source_datasets, list) or not raw_source_datasets:
        raise ValueError('模板清单字段 "source_datasets" 必须是非空数组')
    source_run_ids: set[str] = set()
    source_frame_sha256s: set[str] = set()
    source_perception_sha256s: set[str] = set()
    frame_hashes_by_run: dict[str, dict[int, str]] = {}
    for index, raw_dataset in enumerate(raw_source_datasets):
        field = f"source_datasets[{index}]"
        item = _mapping(raw_dataset, field=field)
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f'模板清单字段 "{field}.run_id" 必须是非空字符串')
        if run_id in source_run_ids:
            raise ValueError(f'模板清单的 source dataset run_id "{run_id}" 不能重复')
        hashes = item.get("frame_sha256s")
        if not isinstance(hashes, list) or not hashes:
            raise ValueError(f'模板清单字段 "{field}.frame_sha256s" 必须是非空数组')
        parsed_frame_hashes = [
            _sha256(frame_hash, field=f"{field}.frame_sha256s") for frame_hash in hashes
        ]
        raw_frame_hashes = item.get("frame_hashes")
        if not isinstance(raw_frame_hashes, list) or not raw_frame_hashes:
            raise ValueError(f'模板清单字段 "{field}.frame_hashes" 必须是非空数组')
        parsed_frame_hashes_by_sequence: dict[int, str] = {}
        content_digest = DatasetContentDigest()
        for frame_index, raw_frame_hash in enumerate(raw_frame_hashes):
            frame_field = f"{field}.frame_hashes[{frame_index}]"
            frame_item = _mapping(raw_frame_hash, field=frame_field)
            sequence = frame_item.get("sequence")
            if type(sequence) is not int or sequence < 0:
                raise ValueError(f'模板清单字段 "{frame_field}.sequence" 必须是非负整数')
            if sequence in parsed_frame_hashes_by_sequence:
                raise ValueError(f'模板清单字段 "{field}.frame_hashes" 的 sequence 不能重复')
            parsed_sha256 = _sha256(
                frame_item.get("sha256"),
                field=f"{frame_field}.sha256",
            )
            content_digest.update_hash(sequence, parsed_sha256)
            parsed_frame_hashes_by_sequence[sequence] = parsed_sha256
        if parsed_frame_hashes != list(parsed_frame_hashes_by_sequence.values()):
            raise ValueError(
                f'模板清单字段 "{field}.frame_sha256s" 必须与 frame_hashes 顺序一致'
            )
        perception_hashes = item.get("perception_sha256s")
        if not isinstance(perception_hashes, list) or not perception_hashes:
            raise ValueError(f'模板清单字段 "{field}.perception_sha256s" 必须是非空数组')
        parsed_perception_hashes = {
            _sha256(value, field=f"{field}.perception_sha256s") for value in perception_hashes
        }
        _sha256(item.get("run_json_sha256"), field=f"{field}.run_json_sha256")
        expected_content_sha256 = _sha256(
            item.get("dataset_content_sha256"),
            field=f"{field}.dataset_content_sha256",
        )
        if content_digest.hexdigest() != expected_content_sha256:
            raise ValueError(
                f'模板清单字段 "{field}.dataset_content_sha256" 与 frame_hashes 不一致'
            )
        _sha256(
            item.get("frame_manifest_sha256"),
            field=f"{field}.frame_manifest_sha256",
        )
        frame_hashes_by_run[run_id] = parsed_frame_hashes_by_sequence
        source_frame_sha256s.update(parsed_frame_hashes)
        source_perception_sha256s.update(parsed_perception_hashes)
        source_run_ids.add(run_id)

    raw_templates = raw.get("templates")
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError('模板清单字段 "templates" 必须是非空数组')
    route_templates: list[RouteTemplate] = []
    root = manifest_path.parent
    for index, raw_template in enumerate(raw_templates):
        field = f"templates[{index}]"
        item = _mapping(raw_template, field=field)
        template_id = item.get("id")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError(f'模板清单字段 "{field}.id" 必须是非空字符串')
        roi_id = item.get("roi_id")
        if not isinstance(roi_id, str) or roi_id not in rois:
            raise ValueError(f'模板清单字段 "{field}.roi_id" 必须引用已定义 ROI')
        route_position = item.get("route_position")
        if not isinstance(route_position, list) or len(route_position) != 2:
            raise ValueError(f'模板清单字段 "{field}.route_position" 必须包含两个数')
        parsed_position = tuple(
            _finite_number(value, field=f"{field}.route_position") for value in route_position
        )
        waypoint_id = item.get("waypoint_id")
        if waypoint_id is not None and (not isinstance(waypoint_id, str) or not waypoint_id):
            raise ValueError(f'模板清单字段 "{field}.waypoint_id" 必须是字符串或 null')
        source_run_id = item.get("source_run_id")
        if not isinstance(source_run_id, str) or not source_run_id:
            raise ValueError(f'模板清单字段 "{field}.source_run_id" 必须是非空字符串')
        if source_run_id not in source_run_ids:
            raise ValueError(f'模板清单字段 "{field}.source_run_id" 必须存在于 source_datasets')
        source_sequence = item.get("source_sequence")
        if type(source_sequence) is not int or source_sequence < 0:
            raise ValueError(f'模板清单字段 "{field}.source_sequence" 必须是非负整数')
        source_frame_sha256 = _sha256(
            item.get("source_frame_sha256"),
            field=f"{field}.source_frame_sha256",
        )
        if frame_hashes_by_run[source_run_id].get(source_sequence) != source_frame_sha256:
            raise ValueError(
                f'模板清单字段 "{field}.source_sequence" 与 source_frame_sha256 '
                "必须匹配同一 source dataset 帧"
            )
        image_path = _safe_asset_path(root, item.get("image"), field=f"{field}.image")
        expected_hash = item.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None
        ):
            raise ValueError(f'模板清单字段 "{field}.sha256" 必须是 64 位十六进制')
        raw_image_bytes = image_path.read_bytes()
        actual_hash = hashlib.sha256(raw_image_bytes).hexdigest()
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                f'模板 "{template_id}" SHA-256 不匹配: '
                f"expected={expected_hash}, actual={actual_hash}"
            )
        encoded_image = np.frombuffer(raw_image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法解码模板图像: {image_path}")
        image_height, image_width = image.shape[:2]
        roi = rois[roi_id]
        if not any(
            2 <= round(image_width * scale) <= roi.width
            and 2 <= round(image_height * scale) <= roi.height
            for scale in scales
        ):
            raise ValueError(
                f'模板 "{template_id}" 没有可用缩放比例：'
                f'缩放后必须至少为 2×2 且不能大于 ROI "{roi_id}"'
            )
        detector = TemplateAnchorDetector(
            label=template_id,
            template=image,
            search_roi=rois[roi_id],
            scales=scales,
            policy=spatial_policy,
            nms_radius_px=nms_radius_px,
        )
        route_templates.append(
            RouteTemplate(
                template_id=template_id,
                detector=detector,
                route_position=parsed_position,
                waypoint_id=waypoint_id,
            )
        )

    observer = TemplateWaypointObserver(
        templates=tuple(route_templates),
        expected_frame_size=frame_size,
        minimum_template_margin=template_margin,
    )
    return TemplateProfile(
        observer=observer,
        frame_size=frame_size,
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        source_run_ids=frozenset(source_run_ids),
        source_frame_sha256s=frozenset(source_frame_sha256s),
        source_perception_sha256s=frozenset(source_perception_sha256s),
        perception_regions=tuple(rois[roi_id] for roi_id in sorted(rois)),
    )
