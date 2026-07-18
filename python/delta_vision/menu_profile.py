"""加载带来源与哈希校验的菜单视觉 Profile。"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CaptureRegion
from .frames import DatasetContentDigest, frame_content_sha256
from .menu_automation import (
    MenuActionKind,
    MenuScene,
    MenuSceneTemplate,
    MenuTransition,
    TemplateMenuSceneObserver,
    VisualMenuController,
)
from .template_matching import MatchDecisionPolicy, TemplateAnchorDetector


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是对象")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    return value


def _integer(
    value: object,
    *,
    field: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} 必须是大于等于 {minimum} 的整数")
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是有限数")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} 必须是有限数") from error
    if not math.isfinite(number) or number < minimum or (
        maximum is not None and number > maximum
    ):
        raise ValueError(f"{field} 超出允许范围")
    return number


def _region(value: object, *, field: str) -> CaptureRegion:
    raw = _mapping(value, field=field)
    left = _integer(raw.get("left"), field=f"{field}.left", minimum=0)
    top = _integer(raw.get("top"), field=f"{field}.top", minimum=0)
    width = _integer(raw.get("width"), field=f"{field}.width", minimum=1)
    height = _integer(raw.get("height"), field=f"{field}.height", minimum=1)
    return CaptureRegion(left=left, top=top, width=width, height=height)


def _scene(value: object, *, field: str) -> MenuScene:
    try:
        return MenuScene(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} 不是已知菜单页面") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝重复字段，避免同一份 Profile 被不同解析器解释成不同配置。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"菜单 Profile JSON 包含重复字段: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"菜单 Profile JSON 包含无效常量: {value}")


@dataclass(frozen=True, slots=True)
class MenuTemplateProvenance:
    template_id: str
    scene: MenuScene
    source_run_id: str
    source_sequence: int
    source_frame_sha256: str
    page_sha256: str
    action_sha256: str | None
    page_content_sha256: str
    action_content_sha256: str | None


@dataclass(frozen=True, slots=True)
class _LoadedDetector:
    detector: TemplateAnchorDetector
    sha256: str
    content_sha256: str
    search_roi: CaptureRegion


@dataclass(frozen=True, slots=True)
class _VerifiedSourceRun:
    run_id: str
    frame_sha256_by_sequence: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class _SourceManifestRecord:
    sequence: int
    image: str


@dataclass(frozen=True, slots=True)
class LoadedMenuProfile:
    profile_id: str
    profile_sha256: str
    source_run_ids: tuple[str, ...]
    template_provenance: tuple[MenuTemplateProvenance, ...]
    observer: TemplateMenuSceneObserver
    transitions: tuple[MenuTransition, ...]
    stop_scenes: frozenset[MenuScene]
    confirmation_frames: int
    maximum_confirmation_span_ms: int
    maximum_action_point_drift_px: float
    maximum_page_point_drift_px: float
    maximum_frame_age_ms: int
    transition_timeout_ms: int

    def create_controller(self) -> VisualMenuController:
        return VisualMenuController(
            transitions=self.transitions,
            confirmation_frames=self.confirmation_frames,
            maximum_confirmation_span_ms=self.maximum_confirmation_span_ms,
            maximum_action_point_drift_px=self.maximum_action_point_drift_px,
            maximum_page_point_drift_px=self.maximum_page_point_drift_px,
            maximum_frame_age_ms=self.maximum_frame_age_ms,
            transition_timeout_ms=self.transition_timeout_ms,
            stop_scenes=self.stop_scenes,
        )


class _MenuProfileLoader:
    def __init__(self, path: Path, payload: bytes) -> None:
        self._path = path
        self._root = path.parent.resolve()
        self._profile_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            text = payload.decode("utf-8")
            parsed = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("菜单 Profile 不是有效 UTF-8 JSON") from error
        self._raw = _mapping(parsed, field="Profile")
        self._image_cache: dict[tuple[Path, str], np.ndarray] = {}

    def _frame_size(self) -> tuple[int, int]:
        raw = self._raw.get("expected_frame_size")
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("expected_frame_size 必须包含宽高")
        return (
            _integer(raw[0], field="expected_frame_size[0]", minimum=1),
            _integer(raw[1], field="expected_frame_size[1]", minimum=1),
        )

    def _source_runs(
        self,
        *,
        frame_size: tuple[int, int],
    ) -> tuple[tuple[str, ...], Mapping[str, _VerifiedSourceRun]]:
        raw_runs = self._raw.get("source_runs")
        if not isinstance(raw_runs, list) or not raw_runs:
            raise ValueError("source_runs 必须是非空数组")
        run_ids: list[str] = []
        verified_runs: dict[str, _VerifiedSourceRun] = {}
        for index, item in enumerate(raw_runs):
            field = f"source_runs[{index}]"
            raw = _mapping(item, field=field)
            run_id = _string(raw.get("run_id"), field=f"{field}.run_id")
            if raw.get("split") != "calibration":
                raise ValueError("菜单模板 source_runs 只能来自 calibration")
            relative = Path(_string(raw.get("directory"), field=f"{field}.directory"))
            if relative.is_absolute():
                raise ValueError(f"{field}.directory 必须位于 Profile 目录")
            dataset_root = (self._root / relative).resolve()
            if not dataset_root.is_relative_to(self._root):
                raise ValueError(f"{field}.directory 不能逃逸 Profile 目录")
            if not dataset_root.is_dir():
                raise ValueError(f"{field}.directory 数据集目录不存在")
            run_path = dataset_root / "run.json"
            manifest_path = dataset_root / "manifest.jsonl"
            if not run_path.is_file() or not manifest_path.is_file():
                raise ValueError(f"{field}.directory 缺少 run.json 或 manifest.jsonl")
            run_payload = run_path.read_bytes()
            manifest_payload = manifest_path.read_bytes()
            expected_run_hash = self._sha256(
                raw.get("run_json_sha256"), field=f"{field}.run_json_sha256"
            )
            expected_manifest_hash = self._sha256(
                raw.get("frame_manifest_sha256"),
                field=f"{field}.frame_manifest_sha256",
            )
            if not hmac.compare_digest(
                hashlib.sha256(run_payload).hexdigest(), expected_run_hash
            ):
                raise ValueError(f"{field}.run_json_sha256 与来源数据集不匹配")
            if not hmac.compare_digest(
                hashlib.sha256(manifest_payload).hexdigest(), expected_manifest_hash
            ):
                raise ValueError(f"{field}.frame_manifest_sha256 与来源数据集不匹配")
            try:
                manifest_text = manifest_payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{field}.manifest 不是有效 UTF-8") from error
            previous_sequence = -1
            previous_captured_at_ns = -1
            manifest_records: list[_SourceManifestRecord] = []
            for line_number, line in enumerate(manifest_text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    parsed_record = json.loads(
                        line,
                        object_pairs_hook=_unique_json_object,
                        parse_constant=_reject_json_constant,
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{field}.manifest 第 {line_number} 行不是有效 JSON"
                    ) from error
                record_field = f"{field}.manifest[{line_number}]"
                record = _mapping(parsed_record, field=record_field)
                sequence = _integer(
                    record.get("sequence"),
                    field=f"{record_field}.sequence",
                    minimum=0,
                )
                captured_at_ns = _integer(
                    record.get("captured_at_ns"),
                    field=f"{record_field}.captured_at_ns",
                    minimum=0,
                )
                if sequence <= previous_sequence:
                    raise ValueError(f"{record_field}.sequence 必须严格递增")
                if captured_at_ns <= previous_captured_at_ns:
                    raise ValueError(f"{record_field}.captured_at_ns 必须严格递增")
                width = _integer(
                    record.get("width"), field=f"{record_field}.width", minimum=1
                )
                height = _integer(
                    record.get("height"), field=f"{record_field}.height", minimum=1
                )
                if (width, height) != frame_size:
                    raise ValueError(f"{record_field} 分辨率与 expected_frame_size 不一致")
                image = _string(record.get("image"), field=f"{record_field}.image")
                _string(record.get("source"), field=f"{record_field}.source")
                metadata = _mapping(
                    record.get("metadata"), field=f"{record_field}.metadata"
                )
                if (
                    metadata.get("run_id") != run_id
                    or metadata.get("dataset_split") != "calibration"
                ):
                    raise ValueError(f"{record_field}.metadata 身份不一致")
                previous_sequence = sequence
                previous_captured_at_ns = captured_at_ns
                manifest_records.append(
                    _SourceManifestRecord(sequence=sequence, image=image)
                )
            try:
                run_text = run_payload.decode("utf-8")
                run_parsed = json.loads(
                    run_text,
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{field}.run.json 不是有效 UTF-8 JSON") from error
            run = _mapping(run_parsed, field=f"{field}.run.json")
            if run.get("run_id") != run_id or run.get("dataset_split") != "calibration":
                raise ValueError(f"{field} 与来源 run.json 的身份或 split 不一致")
            raw_resolution = run.get("resolution")
            if (
                not isinstance(raw_resolution, list)
                or len(raw_resolution) != 2
                or any(type(value) is not int for value in raw_resolution)
                or tuple(raw_resolution) != frame_size
            ):
                raise ValueError(f"{field}.run.json 分辨率与 expected_frame_size 不一致")
            expected_frame_count = _integer(
                run.get("frame_count"),
                field=f"{field}.run.json.frame_count",
                minimum=1,
            )
            content_digest = DatasetContentDigest()
            frame_hashes: dict[int, str] = {}
            for record in manifest_records:
                relative_image = Path(record.image)
                if relative_image.is_absolute():
                    raise ValueError(f"{field}.manifest 图像必须位于数据集目录")
                image_path = (dataset_root / relative_image).resolve()
                if not image_path.is_relative_to(dataset_root):
                    raise ValueError(f"{field}.manifest 图像不能逃逸数据集目录")
                if not image_path.is_file():
                    raise ValueError(f"{field}.manifest 图像文件不存在")
                image_payload = image_path.read_bytes()
                image = cv2.imdecode(
                    np.frombuffer(image_payload, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None or image.size == 0:
                    raise ValueError(f"{field}.manifest 图像无法解码")
                actual_size = (int(image.shape[1]), int(image.shape[0]))
                if actual_size != frame_size:
                    raise ValueError(f"{field} 的来源帧分辨率不一致")
                frame_hashes[record.sequence] = content_digest.update(
                    record.sequence,
                    image,
                )
            if len(frame_hashes) != expected_frame_count:
                raise ValueError(f"{field}.run.json.frame_count 与 manifest 不一致")
            if len(manifest_records) != expected_frame_count:
                raise ValueError(f"{field}.manifest 记录数与 run.json.frame_count 不一致")
            expected_content_hash = self._sha256(
                raw.get("dataset_content_sha256"),
                field=f"{field}.dataset_content_sha256",
            )
            if not hmac.compare_digest(
                content_digest.hexdigest(), expected_content_hash
            ):
                raise ValueError(f"{field}.dataset_content_sha256 与来源帧不匹配")
            run_ids.append(run_id)
            verified_runs[run_id] = _VerifiedSourceRun(
                run_id=run_id,
                frame_sha256_by_sequence=frame_hashes,
            )
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("source_runs.run_id 不能重复")
        return tuple(run_ids), verified_runs

    @staticmethod
    def _sha256(value: object, *, field: str) -> str:
        normalized = _string(value, field=field).lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(f"{field} 必须是 64 位 SHA-256")
        return normalized

    def _image(self, raw_path: object, raw_sha256: object, *, field: str) -> np.ndarray:
        relative = Path(_string(raw_path, field=f"{field}.image"))
        if relative.is_absolute():
            raise ValueError(f"{field}.image 必须位于 Profile 目录")
        candidate = (self._root / relative).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"{field}.image 不能逃逸 Profile 目录")
        if not candidate.is_file():
            raise ValueError(f"{field}.image 文件不存在")
        expected_hash = self._sha256(raw_sha256, field=f"{field}.sha256")
        cache_key = (candidate, expected_hash)
        cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached
        payload = candidate.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(f"{field}.image SHA-256 不匹配")
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise ValueError(f"{field}.image 无法解码")
        decoded.setflags(write=False)
        self._image_cache[cache_key] = decoded
        return decoded

    def _detector(
        self,
        value: object,
        *,
        field: str,
        label: str,
        frame_size: tuple[int, int],
    ) -> _LoadedDetector:
        raw = _mapping(value, field=field)
        search_roi = _region(raw.get("search_roi"), field=f"{field}.search_roi")
        if search_roi.right > frame_size[0] or search_roi.bottom > frame_size[1]:
            raise ValueError(f"{field}.search_roi 超出 expected_frame_size")
        raw_scales = raw.get("scales")
        if not isinstance(raw_scales, list) or not raw_scales:
            raise ValueError(f"{field}.scales 必须是非空数组")
        scales = tuple(
            _finite_number(scale, field=f"{field}.scales", minimum=0.000001)
            for scale in raw_scales
        )
        image = self._image(raw.get("image"), raw.get("sha256"), field=field)
        template_height, template_width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for scale in scales:
            if (
                scale > search_roi.width / template_width
                or scale > search_roi.height / template_height
            ):
                raise ValueError(f"{field}.scale 无法放入 search_roi")
            scaled_width = max(1, round(template_width * scale))
            scaled_height = max(1, round(template_height * scale))
            if scaled_width < 2 or scaled_height < 2:
                raise ValueError(f"{field}.scale 缩放后尺寸必须至少为 2x2")
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            scaled_gray = cv2.resize(
                gray,
                (scaled_width, scaled_height),
                interpolation=interpolation,
            )
            if float(np.std(scaled_gray)) < 1:
                raise ValueError(f"{field}.scale 缩放后纹理不足")
        expected_hash = _string(raw.get("sha256"), field=f"{field}.sha256").lower()
        detector = TemplateAnchorDetector(
            label=label,
            template=image,
            search_roi=search_roi,
            scales=scales,
            policy=MatchDecisionPolicy(
                score_threshold=_finite_number(
                    raw.get("score_threshold"),
                    field=f"{field}.score_threshold",
                    minimum=0.000001,
                    maximum=1,
                ),
                minimum_margin=_finite_number(
                    raw.get("minimum_margin"),
                    field=f"{field}.minimum_margin",
                    minimum=0.000001,
                    maximum=1,
                ),
            ),
            nms_radius_px=_integer(
                raw.get("nms_radius_px"),
                field=f"{field}.nms_radius_px",
                minimum=1,
            ),
        )
        return _LoadedDetector(
            detector=detector,
            sha256=expected_hash,
            content_sha256=frame_content_sha256(image),
            search_roi=search_roi,
        )

    def _templates(
        self,
        *,
        frame_size: tuple[int, int],
        source_runs: Mapping[str, _VerifiedSourceRun],
    ) -> tuple[tuple[MenuSceneTemplate, ...], tuple[MenuTemplateProvenance, ...]]:
        raw_templates = self._raw.get("templates")
        if not isinstance(raw_templates, list) or not raw_templates:
            raise ValueError("templates 必须是非空数组")
        templates: list[MenuSceneTemplate] = []
        provenance: list[MenuTemplateProvenance] = []
        declared_runs = frozenset(source_runs)
        for index, item in enumerate(raw_templates):
            field = f"templates[{index}]"
            raw = _mapping(item, field=field)
            template_id = _string(raw.get("template_id"), field=f"{field}.template_id")
            source_run_id = _string(
                raw.get("source_run_id"), field=f"{field}.source_run_id"
            )
            if source_run_id not in declared_runs:
                raise ValueError(f"{field}.source_run_id 未在 source_runs 声明")
            source_sequence = _integer(
                raw.get("source_sequence"),
                field=f"{field}.source_sequence",
                minimum=0,
            )
            source_frame_sha256 = self._sha256(
                raw.get("source_frame_sha256"),
                field=f"{field}.source_frame_sha256",
            )
            if (
                source_runs[source_run_id].frame_sha256_by_sequence.get(source_sequence)
                != source_frame_sha256
            ):
                raise ValueError(
                    f"{field}.source_sequence 与 source_frame_sha256 "
                    "必须匹配同一来源数据集帧"
                )
            page = self._detector(
                raw.get("page"),
                field=f"{field}.page",
                label=f"{template_id}:page",
                frame_size=frame_size,
            )
            raw_action = raw.get("action")
            raw_action_region = raw.get("action_region")
            if (raw_action is None) != (raw_action_region is None):
                raise ValueError(f"{field}.action 与 action_region 必须同时存在")
            action = (
                None
                if raw_action is None
                else self._detector(
                    raw_action,
                    field=f"{field}.action",
                    label=f"{template_id}:action",
                    frame_size=frame_size,
                )
            )
            action_region = (
                None
                if raw_action_region is None
                else _region(raw_action_region, field=f"{field}.action_region")
            )
            if action_region is not None and (
                action_region.right > frame_size[0]
                or action_region.bottom > frame_size[1]
            ):
                raise ValueError(f"{field}.action_region 超出 expected_frame_size")
            if action is not None and action.content_sha256 == page.content_sha256:
                raise ValueError(f"{field} 的页面锚点和动作锚点必须使用独立图像")
            if action is not None and action_region is not None and not (
                action.search_roi.left <= action_region.left
                and action.search_roi.top <= action_region.top
                and action_region.right <= action.search_roi.right
                and action_region.bottom <= action.search_roi.bottom
            ):
                raise ValueError(f"{field}.action_region 必须位于 action.search_roi 内")
            scene = _scene(raw.get("scene"), field=f"{field}.scene")
            templates.append(
                MenuSceneTemplate(
                    template_id=template_id,
                    scene=scene,
                    page_detector=page.detector,
                    action_detector=None if action is None else action.detector,
                    action_region=action_region,
                )
            )
            provenance.append(
                MenuTemplateProvenance(
                    template_id=template_id,
                    scene=scene,
                    source_run_id=source_run_id,
                    source_sequence=source_sequence,
                    source_frame_sha256=source_frame_sha256,
                    page_sha256=page.sha256,
                    action_sha256=None if action is None else action.sha256,
                    page_content_sha256=page.content_sha256,
                    action_content_sha256=(
                        None if action is None else action.content_sha256
                    ),
                )
            )
        return tuple(templates), tuple(provenance)

    def _transitions(
        self,
        templates: tuple[MenuSceneTemplate, ...],
    ) -> tuple[MenuTransition, ...]:
        raw_transitions = self._raw.get("transitions")
        if not isinstance(raw_transitions, list) or not raw_transitions:
            raise ValueError("transitions 必须是非空数组")
        transitions: list[MenuTransition] = []
        template_scenes = frozenset(template.scene for template in templates)
        actionable_scenes = frozenset(
            template.scene
            for template in templates
            if template.action_detector is not None
        )
        for index, item in enumerate(raw_transitions):
            field = f"transitions[{index}]"
            raw = _mapping(item, field=field)
            try:
                kind = MenuActionKind(raw.get("action_kind"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field}.action_kind 无效") from error
            source = _scene(raw.get("source"), field=f"{field}.source")
            target = _scene(raw.get("target"), field=f"{field}.target")
            if source not in template_scenes:
                raise ValueError(f"{field}.source 没有对应模板")
            if target not in template_scenes:
                raise ValueError(f"{field}.target 没有对应模板")
            if kind is MenuActionKind.CLICK and source not in actionable_scenes:
                raise ValueError(f"{field} 的点击来源页面没有 action 模板")
            transitions.append(
                MenuTransition(
                    source=source,
                    target=target,
                    action_kind=kind,
                    key=raw.get("key"),
                )
            )
        for previous, current in pairwise(transitions):
            if previous.target is not current.source:
                raise ValueError("菜单转换必须首尾连续")
        tokens = tuple((transition.source, transition.target) for transition in transitions)
        if len(set(tokens)) != len(tokens):
            raise ValueError("菜单转换不能包含重复的来源和目标")
        return tuple(transitions)

    def load(self) -> LoadedMenuProfile:
        schema_version = self._raw.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError("不支持的 menu profile schema_version")
        profile_id = _string(self._raw.get("profile_id"), field="profile_id")
        frame_size = self._frame_size()
        source_run_ids, source_runs = self._source_runs(frame_size=frame_size)
        templates, template_provenance = self._templates(
            frame_size=frame_size,
            source_runs=source_runs,
        )
        transitions = self._transitions(templates)
        raw_stop_scenes = self._raw.get("stop_scenes")
        if not isinstance(raw_stop_scenes, list):
            raise ValueError("stop_scenes 必须是数组")
        stop_scenes = frozenset(
            _scene(value, field="stop_scenes") for value in raw_stop_scenes
        )
        template_scenes = frozenset(template.scene for template in templates)
        if any(scene not in template_scenes for scene in stop_scenes):
            raise ValueError("stop_scenes 包含没有对应模板的页面")
        transition_sources = frozenset(
            transition.source for transition in transitions
        )
        if stop_scenes & transition_sources:
            raise ValueError("stop_scenes 不能包含菜单转换的来源页面")
        controller = _mapping(self._raw.get("controller"), field="controller")
        minimum_scene_margin = _finite_number(
            self._raw.get("minimum_scene_margin"),
            field="minimum_scene_margin",
            minimum=0.000001,
            maximum=1,
        )
        loaded = LoadedMenuProfile(
            profile_id=profile_id,
            profile_sha256=self._profile_sha256,
            source_run_ids=source_run_ids,
            template_provenance=template_provenance,
            observer=TemplateMenuSceneObserver(
                templates=templates,
                expected_frame_size=frame_size,
                minimum_scene_margin=minimum_scene_margin,
            ),
            transitions=transitions,
            stop_scenes=stop_scenes,
            confirmation_frames=_integer(
                controller.get("confirmation_frames"),
                field="controller.confirmation_frames",
                minimum=1,
            ),
            maximum_confirmation_span_ms=_integer(
                controller.get("maximum_confirmation_span_ms"),
                field="controller.maximum_confirmation_span_ms",
                minimum=1,
            ),
            maximum_action_point_drift_px=_finite_number(
                controller.get("maximum_action_point_drift_px"),
                field="controller.maximum_action_point_drift_px",
                minimum=0,
            ),
            maximum_page_point_drift_px=_finite_number(
                controller.get("maximum_page_point_drift_px"),
                field="controller.maximum_page_point_drift_px",
                minimum=0,
            ),
            maximum_frame_age_ms=_integer(
                controller.get("maximum_frame_age_ms"),
                field="controller.maximum_frame_age_ms",
                minimum=1,
            ),
            transition_timeout_ms=_integer(
                controller.get("transition_timeout_ms"),
                field="controller.transition_timeout_ms",
                minimum=1,
            ),
        )
        # 在返回前构造一次控制器，确保跨字段约束不会延迟到运行期才失败。
        loaded.create_controller()
        return loaded


def load_menu_profile(path: str | Path) -> LoadedMenuProfile:
    profile_path = Path(path).resolve()
    if not profile_path.is_file():
        raise ValueError("菜单 Profile 文件不存在")
    return _MenuProfileLoader(profile_path, profile_path.read_bytes()).load()
