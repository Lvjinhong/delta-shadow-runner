"""从标准 1920×1080 calibration 数据集生成可移植菜单 Profile。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .frames import DatasetContentDigest, ReplayFrameSource, frame_content_sha256
from .menu_profile import load_menu_profile
from .package_menu_profile import materialize_menu_profile
from .sample_frames import validate_run_id

_FRAME_SIZE = (1920, 1080)
_REGION_KEYS = frozenset({"left", "top", "width", "height"})
_CONTROLLER_KEYS = frozenset(
    {
        "confirmation_frames",
        "maximum_confirmation_span_ms",
        "maximum_action_point_drift_px",
        "maximum_page_point_drift_px",
        "maximum_frame_age_ms",
        "transition_timeout_ms",
    }
)
_TRANSITION_KEYS = frozenset({"source", "target", "action_kind", "key"})


@dataclass(frozen=True, slots=True)
class MenuProfileGenerationResult:
    profile_path: Path
    profile_id: str
    frame_size: tuple[int, int]
    source_run_ids: tuple[str, ...]
    template_count: int
    template_asset_count: int


@dataclass(frozen=True, slots=True)
class _Region:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class _Dataset:
    run_id: str
    directory: Path
    run_json_sha256: str
    frame_manifest_sha256: str
    dataset_content_sha256: str
    frames: Mapping[int, np.ndarray]
    frame_hashes: Mapping[int, str]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 包含无效常量: {value}")


def _load_json(path: Path, *, field: str) -> Any:
    try:
        payload = path.read_bytes()
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} 不是有效 UTF-8 JSON: {path}") from error


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是对象")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{field} 字段不完整或包含未知字段: missing={missing}, extra={unexpected}")


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是有限数")
    number = float(value)
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{field} 超出允许范围")
    return number


def _positive_integer(value: object, *, field: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} 必须是大于等于 {minimum} 的整数")
    return value


def _region(value: object, *, field: str) -> _Region:
    raw = _mapping(value, field=field)
    _exact_keys(raw, set(_REGION_KEYS), field=field)
    region = _Region(
        left=_positive_integer(raw["left"], field=f"{field}.left", minimum=0),
        top=_positive_integer(raw["top"], field=f"{field}.top", minimum=0),
        width=_positive_integer(raw["width"], field=f"{field}.width"),
        height=_positive_integer(raw["height"], field=f"{field}.height"),
    )
    if region.right > _FRAME_SIZE[0] or region.bottom > _FRAME_SIZE[1]:
        raise ValueError(f"{field} 超出 1920x1080 帧边界")
    return region


def _contains(outer: _Region, inner: _Region) -> bool:
    return (
        outer.left <= inner.left
        and outer.top <= inner.top
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"manifest 不是有效 UTF-8: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"manifest 第 {line_number} 行不是有效 JSON") from error
        records.append(_mapping(parsed, field=f"manifest[{line_number}]"))
    if not records:
        raise ValueError("calibration manifest 不能为空")
    return records


def _load_dataset(directory: Path) -> _Dataset:
    root = directory.resolve()
    if directory.is_symlink() or not root.is_dir():
        raise ValueError(f"数据集必须是普通目录: {directory}")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"数据集不能包含符号链接: {candidate}")
    run_path = root / "run.json"
    manifest_path = root / "manifest.jsonl"
    if not run_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"数据集缺少 run.json 或 manifest.jsonl: {root}")
    run = _mapping(_load_json(run_path, field="run.json"), field="run.json")
    run_id = validate_run_id(run.get("run_id"))
    if run.get("dataset_split") != "calibration":
        raise ValueError(f"数据集 {run_id} 必须来自 calibration split")
    if run.get("resolution") != list(_FRAME_SIZE):
        raise ValueError(f"数据集 {run_id} 分辨率必须是 1920x1080")
    frame_count = _positive_integer(run.get("frame_count"), field="run.json.frame_count")
    manifest = _load_manifest(manifest_path)
    replay_frames = list(ReplayFrameSource(root))
    if len(manifest) != frame_count or len(replay_frames) != frame_count:
        raise ValueError(f"数据集 {run_id} 的 frame_count 与 manifest 不一致")
    digest = DatasetContentDigest()
    frames: dict[int, np.ndarray] = {}
    frame_hashes: dict[int, str] = {}
    for index, (record, frame) in enumerate(zip(manifest, replay_frames, strict=True)):
        if frame.image.shape != (_FRAME_SIZE[1], _FRAME_SIZE[0], 3):
            raise ValueError(f"数据集 {run_id} 第 {index} 帧分辨率必须是 1920x1080")
        if record.get("sequence") != frame.sequence:
            raise ValueError(f"数据集 {run_id} 第 {index} 帧序号不一致")
        metadata = _mapping(record.get("metadata"), field=f"manifest[{index}].metadata")
        if metadata.get("run_id") != run_id or metadata.get("dataset_split") != "calibration":
            raise ValueError(f"数据集 {run_id} 第 {index} 帧身份不一致")
        if frame.sequence in frames:
            raise ValueError(f"数据集 {run_id} 的帧序号重复")
        frames[frame.sequence] = frame.image
        frame_hashes[frame.sequence] = digest.update(frame.sequence, frame.image)
    return _Dataset(
        run_id=run_id,
        directory=root,
        run_json_sha256=hashlib.sha256(run_path.read_bytes()).hexdigest(),
        frame_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        dataset_content_sha256=digest.hexdigest(),
        frames=frames,
        frame_hashes=frame_hashes,
    )


def _load_datasets(directories: Sequence[str | Path]) -> dict[str, _Dataset]:
    if not directories:
        raise ValueError("至少需要一个 calibration 数据集")
    datasets: dict[str, _Dataset] = {}
    for value in directories:
        dataset = _load_dataset(Path(value))
        if dataset.run_id in datasets:
            raise ValueError(f"数据集 run_id 重复: {dataset.run_id}")
        datasets[dataset.run_id] = dataset
    return datasets


def _detector_defaults(value: object) -> dict[str, object]:
    raw = _mapping(value, field="detector_defaults")
    _exact_keys(
        raw,
        {"scales", "score_threshold", "minimum_margin", "nms_radius_px"},
        field="detector_defaults",
    )
    scales = raw["scales"]
    if not isinstance(scales, list) or not scales:
        raise ValueError("detector_defaults.scales 必须是非空数组")
    parsed_scales = [
        _finite_number(scale, field="detector_defaults.scales", minimum=0.000001)
        for scale in scales
    ]
    return {
        "scales": parsed_scales,
        "score_threshold": _finite_number(
            raw["score_threshold"],
            field="detector_defaults.score_threshold",
            minimum=0.000001,
            maximum=1,
        ),
        "minimum_margin": _finite_number(
            raw["minimum_margin"],
            field="detector_defaults.minimum_margin",
            minimum=0.000001,
            maximum=1,
        ),
        "nms_radius_px": _positive_integer(
            raw["nms_radius_px"], field="detector_defaults.nms_radius_px"
        ),
    }


def _write_crop(frame: np.ndarray, crop: _Region, destination: Path, *, field: str) -> str:
    image = frame[crop.top : crop.bottom, crop.left : crop.right]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 1:
        raise ValueError(f"{field}.crop 纹理不足")
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"无法编码 {field}.crop")
    raw = payload.tobytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _build_detector(
    raw_value: object,
    *,
    field: str,
    frame: np.ndarray,
    defaults: Mapping[str, object],
    asset_path: Path,
    asset_reference: str,
) -> tuple[dict[str, object], _Region]:
    raw = _mapping(raw_value, field=field)
    _exact_keys(raw, {"crop", "search_roi"}, field=field)
    crop = _region(raw["crop"], field=f"{field}.crop")
    search_roi = _region(raw["search_roi"], field=f"{field}.search_roi")
    if not _contains(search_roi, crop):
        raise ValueError(f"{field}.crop 必须位于 search_roi 内")
    sha256 = _write_crop(frame, crop, asset_path, field=field)
    return (
        {
            "image": asset_reference,
            "sha256": sha256,
            "search_roi": search_roi.as_dict(),
            **defaults,
        },
        crop,
    )


def _build_action(
    raw_value: object,
    *,
    field: str,
    frame: np.ndarray,
    defaults: Mapping[str, object],
    asset_path: Path,
    asset_reference: str,
) -> tuple[dict[str, object], _Region, _Region]:
    raw = _mapping(raw_value, field=field)
    _exact_keys(raw, {"crop", "search_roi", "action_region"}, field=field)
    detector, crop = _build_detector(
        {"crop": raw["crop"], "search_roi": raw["search_roi"]},
        field=field,
        frame=frame,
        defaults=defaults,
        asset_path=asset_path,
        asset_reference=asset_reference,
    )
    action_region = _region(raw["action_region"], field=f"{field}.action_region")
    search_roi = _region(raw["search_roi"], field=f"{field}.search_roi")
    if not _contains(search_roi, action_region):
        raise ValueError(f"{field}.action_region 必须位于 search_roi 内")
    if not _contains(action_region, crop):
        raise ValueError(f"{field}.crop 必须位于 action_region 内")
    return detector, action_region, crop


def _build_templates(
    raw_templates: object,
    *,
    datasets: Mapping[str, _Dataset],
    defaults: Mapping[str, object],
    root: Path,
) -> tuple[list[dict[str, object]], set[str]]:
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("templates 必须是非空数组")
    templates: list[dict[str, object]] = []
    used_runs: set[str] = set()
    seen_ids: set[str] = set()
    for index, value in enumerate(raw_templates):
        field = f"templates[{index}]"
        raw = _mapping(value, field=field)
        required = {"template_id", "scene", "source_run_id", "source_sequence", "page"}
        allowed = required | {"action"}
        if not required <= set(raw) or not set(raw) <= allowed:
            raise ValueError(f"{field} 字段不完整或包含未知字段")
        template_id = validate_run_id(raw["template_id"])
        if template_id in seen_ids:
            raise ValueError(f"template_id 重复: {template_id}")
        seen_ids.add(template_id)
        run_id = validate_run_id(raw["source_run_id"])
        dataset = datasets.get(run_id)
        if dataset is None:
            raise ValueError(f"{field}.source_run_id 没有对应数据集: {run_id}")
        sequence = _positive_integer(
            raw["source_sequence"], field=f"{field}.source_sequence", minimum=0
        )
        frame = dataset.frames.get(sequence)
        if frame is None:
            raise ValueError(f"{field}.source_sequence 序号不存在: {sequence}")
        page_ref = f"templates/{template_id}-page.png"
        page, page_crop = _build_detector(
            raw["page"],
            field=f"{field}.page",
            frame=frame,
            defaults=defaults,
            asset_path=root / page_ref,
            asset_reference=page_ref,
        )
        action_raw = raw.get("action")
        action: dict[str, object] | None = None
        action_region: dict[str, int] | None = None
        if action_raw is not None:
            action_ref = f"templates/{template_id}-action.png"
            action, parsed_region, action_crop = _build_action(
                action_raw,
                field=f"{field}.action",
                frame=frame,
                defaults=defaults,
                asset_path=root / action_ref,
                asset_reference=action_ref,
            )
            if page_crop == action_crop or frame_content_sha256(
                frame[page_crop.top : page_crop.bottom, page_crop.left : page_crop.right]
            ) == frame_content_sha256(
                frame[
                    action_crop.top : action_crop.bottom,
                    action_crop.left : action_crop.right,
                ]
            ):
                raise ValueError(f"{field} 的 page 与 action 必须使用独立图像")
            action_region = parsed_region.as_dict()
        scene = raw["scene"]
        if not isinstance(scene, str) or not scene:
            raise ValueError(f"{field}.scene 必须是非空字符串")
        templates.append(
            {
                "template_id": template_id,
                "scene": scene,
                "source_run_id": run_id,
                "source_sequence": sequence,
                "source_frame_sha256": dataset.frame_hashes[sequence],
                "page": page,
                "action": action,
                "action_region": action_region,
            }
        )
        used_runs.add(run_id)
    return templates, used_runs


def _validate_click_transitions(
    raw_transitions: object,
    templates: Sequence[Mapping[str, object]],
) -> None:
    if not isinstance(raw_transitions, list) or not raw_transitions:
        raise ValueError("transitions 必须是非空数组")
    for index, value in enumerate(raw_transitions):
        raw = _mapping(value, field=f"transitions[{index}]")
        if raw.get("action_kind") != "click":
            continue
        source = raw.get("source")
        source_templates = [template for template in templates if template.get("scene") == source]
        if not source_templates:
            raise ValueError(f"transitions[{index}] 的点击来源页面没有模板")
        if any(template.get("action") is None for template in source_templates):
            raise ValueError(f"transitions[{index}] 的点击来源 scene 每个模板都必须配置 action")


def _source_runs(datasets: Mapping[str, _Dataset], used_runs: set[str]) -> list[dict[str, str]]:
    if set(datasets) != used_runs:
        unused = sorted(set(datasets) - used_runs)
        raise ValueError(f"存在未被模板引用的数据集: {unused}")
    return [
        {
            "run_id": run_id,
            "split": "calibration",
            "directory": f"datasets/{run_id}",
            "run_json_sha256": datasets[run_id].run_json_sha256,
            "frame_manifest_sha256": datasets[run_id].frame_manifest_sha256,
            "dataset_content_sha256": datasets[run_id].dataset_content_sha256,
        }
        for run_id in sorted(datasets)
    ]


def _build_source_profile(
    spec: dict[str, Any],
    datasets: Mapping[str, _Dataset],
    root: Path,
) -> Path:
    defaults = _detector_defaults(spec["detector_defaults"])
    templates, used_runs = _build_templates(
        spec["templates"], datasets=datasets, defaults=defaults, root=root
    )
    _validate_click_transitions(spec["transitions"], templates)
    for run_id in sorted(datasets):
        shutil.copytree(datasets[run_id].directory, root / "datasets" / run_id)
    profile = {
        "schema_version": spec["schema_version"],
        "profile_id": spec["profile_id"],
        "expected_frame_size": spec["expected_frame_size"],
        "minimum_scene_margin": spec["minimum_scene_margin"],
        "controller": spec["controller"],
        "source_runs": _source_runs(datasets, used_runs),
        "templates": templates,
        "transitions": spec["transitions"],
        "stop_scenes": spec["stop_scenes"],
    }
    path = root / "menu.json"
    path.write_text(
        json.dumps(
            profile,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _load_spec(path: Path) -> dict[str, Any]:
    spec = _mapping(_load_json(path, field="菜单 Profile spec"), field="菜单 Profile spec")
    _exact_keys(
        spec,
        {
            "schema_version",
            "profile_id",
            "expected_frame_size",
            "minimum_scene_margin",
            "detector_defaults",
            "controller",
            "templates",
            "transitions",
            "stop_scenes",
        },
        field="菜单 Profile spec",
    )
    if spec["schema_version"] != 1:
        raise ValueError("不支持的 spec schema_version")
    validate_run_id(spec["profile_id"])
    if spec["expected_frame_size"] != list(_FRAME_SIZE):
        raise ValueError("expected_frame_size 必须固定为 [1920, 1080]")
    _finite_number(
        spec["minimum_scene_margin"],
        field="minimum_scene_margin",
        minimum=0.000001,
        maximum=1,
    )
    controller = _mapping(spec["controller"], field="controller")
    _exact_keys(controller, set(_CONTROLLER_KEYS), field="controller")
    transitions = spec["transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("transitions 必须是非空数组")
    for index, value in enumerate(transitions):
        transition = _mapping(value, field=f"transitions[{index}]")
        _exact_keys(
            transition,
            set(_TRANSITION_KEYS),
            field=f"transitions[{index}]",
        )
    if not isinstance(spec["stop_scenes"], list):
        raise ValueError("stop_scenes 必须是数组")
    return spec


def generate_menu_profile(
    *,
    spec_path: str | Path,
    dataset_directories: Sequence[str | Path],
    output_directory: str | Path,
) -> MenuProfileGenerationResult:
    """校验来源、裁剪锚点，并以 no-replace 方式发布完整 Profile。"""

    output = Path(output_directory).resolve()
    if output.exists():
        raise ValueError(f"输出目录已经存在: {output}")
    spec = _load_spec(Path(spec_path).resolve())
    datasets = _load_datasets(dataset_directories)
    with tempfile.TemporaryDirectory(prefix="delta-menu-profile-") as temporary:
        source_profile = _build_source_profile(spec, datasets, Path(temporary))
        # 先让正式 loader 验证临时 Profile，再调用已有原子 bundle 发布路径。
        load_menu_profile(source_profile)
        bundle = materialize_menu_profile(source_profile, output)
    loaded = load_menu_profile(bundle.profile_path)
    return MenuProfileGenerationResult(
        profile_path=bundle.profile_path,
        profile_id=loaded.profile_id,
        frame_size=loaded.frame_size,
        source_run_ids=loaded.source_run_ids,
        template_count=len(loaded.template_provenance),
        template_asset_count=bundle.template_asset_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 1920x1080 calibration 数据集生成菜单 Profile")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = generate_menu_profile(
            spec_path=args.spec,
            dataset_directories=args.dataset,
            output_directory=args.output,
        )
    except Exception as error:
        print(f"菜单 Profile 生成失败: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "complete",
                "profile": str(result.profile_path),
                "profile_id": result.profile_id,
                "frame_size": list(result.frame_size),
                "source_run_ids": list(result.source_run_ids),
                "template_count": result.template_count,
                "template_asset_count": result.template_asset_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
