"""只根据截图确认页面，并生成一次性的菜单动作意图。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from statistics import median

from .config import CaptureRegion
from .frames import CapturedFrame
from .template_matching import TemplateAnchorDetector, TemplateMatchObservation


class MenuScene(StrEnum):
    """零成本进图流程涉及的最小页面集合。"""

    UNKNOWN = "unknown"
    LOBBY = "lobby"
    STRATEGY_BOARD = "strategy_board"
    ZERO_DAM_READY = "zero_dam_ready"
    IN_MATCH = "in_match"
    DEATH_SUMMARY = "death_summary"
    POST_MATCH = "post_match"


class SceneDecisionReason(StrEnum):
    ACCEPTED = "accepted"
    BELOW_THRESHOLD = "below_threshold"
    AMBIGUOUS = "ambiguous"
    FRAME_SIZE_MISMATCH = "frame_size_mismatch"


class MenuActionKind(StrEnum):
    CLICK = "click"
    KEY = "key"


class MenuControllerStatus(StrEnum):
    OBSERVING = "observing"
    WAITING_FOR_TRANSITION = "waiting_for_transition"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class MenuSceneTemplate:
    """页面锚点与动作锚点必须独立通过，避免仅凭背景固定点击。"""

    template_id: str
    scene: MenuScene
    page_detector: TemplateAnchorDetector
    action_detector: TemplateAnchorDetector | None
    action_region: CaptureRegion | None

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("菜单模板 ID 不能为空")
        if not isinstance(self.scene, MenuScene) or self.scene is MenuScene.UNKNOWN:
            raise ValueError("不能为 UNKNOWN 配置页面模板")
        if self.action_detector is None and self.action_region is not None:
            raise ValueError("没有动作模板时不能配置动作区域")
        if self.action_detector is not None and self.action_region is None:
            raise ValueError("动作模板必须配置显式动作区域")
        if self.action_detector is not None and self.page_detector is self.action_detector:
            raise ValueError("页面锚点和动作锚点必须独立")


@dataclass(frozen=True, slots=True)
class SceneObservation:
    frame_sequence: int
    captured_at_ns: int
    scene: MenuScene
    candidate_scene: MenuScene | None
    confidence: float
    runner_up_confidence: float
    accepted: bool
    reason: SceneDecisionReason
    action_accepted: bool
    action_point: tuple[float, float] | None
    page_point: tuple[float, float] | None
    page_template_id: str | None

    @staticmethod
    def _valid_point(point: tuple[float, float] | None) -> bool:
        return bool(
            point is not None
            and len(point) == 2
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
        )

    def __post_init__(self) -> None:
        if type(self.frame_sequence) is not int or self.frame_sequence < 0:
            raise ValueError("截图序号必须是非负整数")
        if type(self.captured_at_ns) is not int or self.captured_at_ns < 0:
            raise ValueError("截图时间戳必须是非负整数")
        if not isinstance(self.scene, MenuScene) or (
            self.candidate_scene is not None
            and not isinstance(self.candidate_scene, MenuScene)
        ):
            raise ValueError("页面状态必须使用 MenuScene")
        if (
            not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
            or not math.isfinite(self.runner_up_confidence)
            or not 0 <= self.runner_up_confidence <= 1
        ):
            raise ValueError("页面置信度必须位于 0 到 1 之间")
        if type(self.accepted) is not bool or type(self.action_accepted) is not bool:
            raise ValueError("页面和动作接受状态必须是布尔值")
        if self.accepted:
            if (
                self.scene is MenuScene.UNKNOWN
                or self.candidate_scene is not self.scene
                or self.reason is not SceneDecisionReason.ACCEPTED
                or not isinstance(self.page_template_id, str)
                or not self.page_template_id
            ):
                raise ValueError("已接受页面的状态、原因和模板必须一致")
            if not self._valid_point(self.page_point):
                raise ValueError("已接受页面必须包含有限页面锚点")
        elif (
            self.scene is not MenuScene.UNKNOWN
            or self.reason is SceneDecisionReason.ACCEPTED
            or self.page_point is not None
        ):
            raise ValueError("已拒绝页面不能携带可执行页面状态")
        if self.action_accepted != (self.action_point is not None):
            raise ValueError("动作接受状态必须与动作锚点同时存在")
        if self.action_point is not None and not self._valid_point(self.action_point):
            raise ValueError("动作锚点必须包含两个有限坐标")
        if not self.accepted and self.action_accepted:
            raise ValueError("页面未接受时不能接受动作锚点")


@dataclass(frozen=True, slots=True)
class MenuTransition:
    source: MenuScene
    target: MenuScene
    action_kind: MenuActionKind
    key: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, MenuScene)
            or not isinstance(self.target, MenuScene)
            or self.source is MenuScene.UNKNOWN
            or self.target is MenuScene.UNKNOWN
        ):
            raise ValueError("菜单转换不能以 UNKNOWN 为端点")
        if not isinstance(self.action_kind, MenuActionKind):
            raise ValueError("菜单转换 action_kind 必须是 MenuActionKind")
        if self.source is self.target:
            raise ValueError("菜单转换的来源和目标页面不能相同")
        if self.action_kind is MenuActionKind.KEY:
            if not isinstance(self.key, str) or not self.key:
                raise ValueError("按键菜单动作必须配置非空 key")
        elif self.key is not None:
            raise ValueError("点击菜单动作不能配置 key")


@dataclass(frozen=True, slots=True)
class MenuAction:
    kind: MenuActionKind
    source: MenuScene
    expected_target: MenuScene
    position: tuple[int, int] | None = None
    key: str | None = None
    expires_at_ns: int = 0

    def __post_init__(self) -> None:
        if type(self.expires_at_ns) is not int or self.expires_at_ns <= 0:
            raise ValueError("菜单动作必须包含正整数过期时间")
        if self.kind is MenuActionKind.CLICK:
            if (
                self.position is None
                or len(self.position) != 2
                or any(type(value) is not int for value in self.position)
                or self.key is not None
            ):
                raise ValueError("点击动作必须只包含整数坐标")
        elif self.kind is MenuActionKind.KEY:
            if self.position is not None or not isinstance(self.key, str) or not self.key:
                raise ValueError("按键动作必须只包含非空 key")
        else:
            raise ValueError("菜单动作 kind 必须是 MenuActionKind")


@dataclass(frozen=True, slots=True)
class MenuControllerSnapshot:
    status: MenuControllerStatus
    transition_index: int
    action: MenuAction | None
    reason: str | None
    observed_scene: MenuScene
    observed_at_ns: int


@dataclass(frozen=True, slots=True)
class _PageCandidate:
    template: MenuSceneTemplate
    match: TemplateMatchObservation


class TemplateMenuSceneObserver:
    """在固定分辨率下选择唯一页面，并单独检测可点击动作锚点。"""

    def __init__(
        self,
        *,
        templates: tuple[MenuSceneTemplate, ...],
        expected_frame_size: tuple[int, int],
        minimum_scene_margin: float,
    ) -> None:
        if not templates:
            raise ValueError("菜单页面模板不能为空")
        template_ids = tuple(template.template_id for template in templates)
        if len(set(template_ids)) != len(template_ids):
            raise ValueError("菜单模板 ID 不能重复")
        if (
            len(expected_frame_size) != 2
            or any(type(value) is not int or value <= 0 for value in expected_frame_size)
        ):
            raise ValueError("期望帧分辨率必须包含两个正整数")
        if (
            not math.isfinite(minimum_scene_margin)
            or not 0 <= minimum_scene_margin <= 1
        ):
            raise ValueError("页面最佳与次佳差值必须位于 0 到 1 之间")
        self._templates = templates
        self._expected_frame_size = expected_frame_size
        self._minimum_scene_margin = minimum_scene_margin

    @staticmethod
    def _unknown(
        frame: CapturedFrame,
        *,
        reason: SceneDecisionReason,
        candidate_scene: MenuScene | None = None,
        confidence: float = 0,
        runner_up_confidence: float = 0,
        page_template_id: str | None = None,
    ) -> SceneObservation:
        return SceneObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            scene=MenuScene.UNKNOWN,
            candidate_scene=candidate_scene,
            confidence=confidence,
            runner_up_confidence=runner_up_confidence,
            accepted=False,
            reason=reason,
            action_accepted=False,
            action_point=None,
            page_point=None,
            page_template_id=page_template_id,
        )

    def observe(self, frame: CapturedFrame) -> SceneObservation:
        actual_size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
        if actual_size != self._expected_frame_size:
            return self._unknown(
                frame,
                reason=SceneDecisionReason.FRAME_SIZE_MISMATCH,
            )

        matches = tuple(
            _PageCandidate(template, template.page_detector.detect(frame.image))
            for template in self._templates
        )
        best_by_scene: dict[MenuScene, _PageCandidate] = {}
        for candidate in matches:
            previous = best_by_scene.get(candidate.template.scene)
            if previous is None or (
                candidate.match.confidence,
                candidate.template.template_id,
            ) > (
                previous.match.confidence,
                previous.template.template_id,
            ):
                best_by_scene[candidate.template.scene] = candidate
        ranked = sorted(
            best_by_scene.values(),
            key=lambda candidate: (
                -candidate.match.confidence,
                candidate.template.template_id,
            ),
        )
        best = ranked[0]
        runner_up_confidence = 0 if len(ranked) == 1 else ranked[1].match.confidence
        candidate_scene = best.template.scene
        if not best.match.accepted:
            reason = (
                SceneDecisionReason.AMBIGUOUS
                if best.match.reason == "ambiguous"
                else SceneDecisionReason.BELOW_THRESHOLD
            )
            return self._unknown(
                frame,
                reason=reason,
                candidate_scene=candidate_scene,
                confidence=best.match.confidence,
                runner_up_confidence=runner_up_confidence,
                page_template_id=best.template.template_id,
            )
        if best.match.confidence - runner_up_confidence < self._minimum_scene_margin:
            return self._unknown(
                frame,
                reason=SceneDecisionReason.AMBIGUOUS,
                candidate_scene=candidate_scene,
                confidence=best.match.confidence,
                runner_up_confidence=runner_up_confidence,
                page_template_id=best.template.template_id,
            )

        action_match = (
            None
            if best.template.action_detector is None
            else best.template.action_detector.detect(frame.image)
        )
        action_point = (
            None
            if action_match is None or not action_match.accepted
            else action_match.centroid
        )
        action_region = best.template.action_region
        action_accepted = bool(
            action_point is not None
            and action_region is not None
            and action_region.left <= action_point[0] < action_region.right
            and action_region.top <= action_point[1] < action_region.bottom
        )
        if not action_accepted:
            action_point = None
        return SceneObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            scene=candidate_scene,
            candidate_scene=candidate_scene,
            confidence=best.match.confidence,
            runner_up_confidence=runner_up_confidence,
            accepted=True,
            reason=SceneDecisionReason.ACCEPTED,
            action_accepted=action_accepted,
            action_point=action_point,
            page_point=best.match.centroid,
            page_template_id=best.template.template_id,
        )


class VisualMenuController:
    """确认来源页面后只发一次动作，并等待目标页面作为后置条件。"""

    def __init__(
        self,
        *,
        transitions: tuple[MenuTransition, ...],
        confirmation_frames: int,
        maximum_confirmation_span_ms: int,
        maximum_action_point_drift_px: float,
        maximum_page_point_drift_px: float,
        maximum_frame_age_ms: int,
        transition_timeout_ms: int,
        stop_scenes: frozenset[MenuScene],
    ) -> None:
        if not transitions:
            raise ValueError("菜单转换不能为空")
        for previous, current in pairwise(transitions):
            if previous.target is not current.source:
                raise ValueError("菜单转换必须首尾连续")
        if type(confirmation_frames) is not int or confirmation_frames <= 0:
            raise ValueError("确认帧数必须是正整数")
        if (
            type(maximum_confirmation_span_ms) is not int
            or maximum_confirmation_span_ms <= 0
        ):
            raise ValueError("确认时间窗口必须是正整数毫秒")
        if (
            not math.isfinite(maximum_action_point_drift_px)
            or maximum_action_point_drift_px < 0
        ):
            raise ValueError("动作锚点漂移必须是非负有限数")
        if (
            not math.isfinite(maximum_page_point_drift_px)
            or maximum_page_point_drift_px < 0
        ):
            raise ValueError("页面锚点漂移必须是非负有限数")
        if type(maximum_frame_age_ms) is not int or maximum_frame_age_ms <= 0:
            raise ValueError("最大帧龄必须是正整数毫秒")
        if type(transition_timeout_ms) is not int or transition_timeout_ms <= 0:
            raise ValueError("页面转换超时必须是正整数毫秒")
        if (
            not isinstance(stop_scenes, frozenset)
            or any(not isinstance(scene, MenuScene) for scene in stop_scenes)
            or MenuScene.UNKNOWN in stop_scenes
        ):
            raise ValueError("UNKNOWN 不能作为确认停止页面")

        self._transitions = transitions
        self._confirmation_frames = confirmation_frames
        self._maximum_confirmation_span_ns = maximum_confirmation_span_ms * 1_000_000
        self._maximum_action_point_drift_px = maximum_action_point_drift_px
        self._maximum_page_point_drift_px = maximum_page_point_drift_px
        self._maximum_frame_age_ns = maximum_frame_age_ms * 1_000_000
        self._transition_timeout_ns = transition_timeout_ms * 1_000_000
        self._stop_scenes = stop_scenes
        self._status = MenuControllerStatus.OBSERVING
        self._transition_index = 0
        self._reason: str | None = None
        self._last_sequence: int | None = None
        self._last_captured_at_ns: int | None = None
        self._streak: list[SceneObservation] = []
        self._pending_since_ns: int | None = None
        self._last_observed_scene = MenuScene.UNKNOWN
        self._last_observed_at_ns = 0

    def _snapshot(self, *, action: MenuAction | None = None) -> MenuControllerSnapshot:
        return MenuControllerSnapshot(
            status=self._status,
            transition_index=self._transition_index,
            action=action,
            reason=self._reason,
            observed_scene=self._last_observed_scene,
            observed_at_ns=self._last_observed_at_ns,
        )

    def _stop(self, reason: str) -> MenuControllerSnapshot:
        self._status = MenuControllerStatus.STOPPED
        self._reason = reason
        self._streak.clear()
        return self._snapshot()

    def stop(self, reason: str) -> MenuControllerSnapshot:
        """由外层运行时显式终止，并保留首次终态原因。"""

        if not isinstance(reason, str) or not reason:
            raise ValueError("停止原因必须是非空字符串")
        if self._status in {
            MenuControllerStatus.STOPPED,
            MenuControllerStatus.COMPLETED,
        }:
            return self._snapshot()
        return self._stop(reason)

    def _record_streak(self, observation: SceneObservation) -> None:
        if not observation.accepted or observation.scene is MenuScene.UNKNOWN:
            self._streak.clear()
            return
        if self._streak:
            previous = self._streak[-1]
            if (
                previous.scene is not observation.scene
                or previous.page_template_id != observation.page_template_id
            ):
                self._streak.clear()
        self._streak.append(observation)
        if (
            observation.captured_at_ns - self._streak[0].captured_at_ns
            > self._maximum_confirmation_span_ns
        ):
            self._streak[:] = [observation]

    def _confirmed_scene(self) -> MenuScene | None:
        if len(self._streak) < self._confirmation_frames:
            return None
        recent = self._streak[-self._confirmation_frames :]
        scene = recent[0].scene
        if any(observation.scene is not scene for observation in recent):
            return None
        template_id = recent[0].page_template_id
        page_points = tuple(observation.page_point for observation in recent)
        if (
            template_id is None
            or any(observation.page_template_id != template_id for observation in recent)
            or any(point is None for point in page_points)
        ):
            self._streak[:] = [recent[-1]]
            return None
        concrete_points = tuple(point for point in page_points if point is not None)
        if any(
            math.dist(first, second) > self._maximum_page_point_drift_px
            for index, first in enumerate(concrete_points)
            for second in concrete_points[index + 1 :]
        ):
            self._streak[:] = [recent[-1]]
            return None
        return scene

    def _stable_action_position(self) -> tuple[int, int] | None:
        recent = self._streak[-self._confirmation_frames :]
        points = tuple(observation.action_point for observation in recent)
        if any(not observation.action_accepted for observation in recent) or any(
            point is None for point in points
        ):
            return None
        concrete_points = tuple(point for point in points if point is not None)
        if any(
            math.dist(first, second) > self._maximum_action_point_drift_px
            for index, first in enumerate(concrete_points)
            for second in concrete_points[index + 1 :]
        ):
            self._streak[:] = [recent[-1]]
            return None
        return (
            round(median(point[0] for point in concrete_points)),
            round(median(point[1] for point in concrete_points)),
        )

    def step(
        self,
        observation: SceneObservation,
        *,
        now_ns: int,
    ) -> MenuControllerSnapshot:
        if self._status in {
            MenuControllerStatus.STOPPED,
            MenuControllerStatus.COMPLETED,
        }:
            return self._snapshot()
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("当前时钟必须是非负整数纳秒")
        frame_age_ns = now_ns - observation.captured_at_ns
        if frame_age_ns < 0:
            return self._stop("当前时钟早于截图时间")
        if frame_age_ns > self._maximum_frame_age_ns:
            return self._stop("截图超过最大允许帧龄")
        if (
            self._last_sequence is not None
            and (
                observation.frame_sequence <= self._last_sequence
                or observation.captured_at_ns <= self._last_captured_at_ns
            )
        ):
            return self._stop("截图序号或时间戳没有严格递增")

        self._last_sequence = observation.frame_sequence
        self._last_captured_at_ns = observation.captured_at_ns
        self._last_observed_scene = observation.scene
        self._last_observed_at_ns = observation.captured_at_ns
        if (
            self._pending_since_ns is not None
            and observation.captured_at_ns - self._pending_since_ns
            >= self._transition_timeout_ns
        ):
            return self._stop("等待目标页面超时")

        self._record_streak(observation)
        confirmed_scene = self._confirmed_scene()
        if confirmed_scene in self._stop_scenes:
            return self._stop(f"检测到停止页面: {confirmed_scene}")

        transition = self._transitions[self._transition_index]
        if self._status is MenuControllerStatus.WAITING_FOR_TRANSITION:
            if confirmed_scene is transition.target:
                self._transition_index += 1
                self._pending_since_ns = None
                self._streak.clear()
                self._reason = None
                if self._transition_index == len(self._transitions):
                    self._status = MenuControllerStatus.COMPLETED
                else:
                    self._status = MenuControllerStatus.OBSERVING
                return self._snapshot()
            if confirmed_scene not in {None, transition.source}:
                return self._stop(
                    f"等待期间出现意外页面: {confirmed_scene}"
                )
            return self._snapshot()

        if confirmed_scene is None:
            return self._snapshot()
        if confirmed_scene is not transition.source:
            return self._stop(
                f"预期页面 {transition.source}，实际确认 {confirmed_scene}"
            )

        if transition.action_kind is MenuActionKind.CLICK:
            position = self._stable_action_position()
            if position is None:
                return self._snapshot()
            action = MenuAction(
                kind=MenuActionKind.CLICK,
                source=transition.source,
                expected_target=transition.target,
                position=position,
                expires_at_ns=(
                    observation.captured_at_ns + self._maximum_frame_age_ns
                ),
            )
        else:
            action = MenuAction(
                kind=MenuActionKind.KEY,
                source=transition.source,
                expected_target=transition.target,
                key=transition.key,
                expires_at_ns=(
                    observation.captured_at_ns + self._maximum_frame_age_ns
                ),
            )
        self._status = MenuControllerStatus.WAITING_FOR_TRANSITION
        self._pending_since_ns = observation.captured_at_ns
        self._reason = None
        self._streak.clear()
        return self._snapshot(action=action)
