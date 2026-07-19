"""保险箱清理的纯状态机；视觉观察与实际输入由外层注入。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class WarehouseScene(StrEnum):
    UNKNOWN = "unknown"
    BASE = "base"
    WAREHOUSE = "warehouse"


class SafeSlotState(StrEnum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    OCCUPIED = "occupied"


class CleanupIntentKind(StrEnum):
    OPEN_WAREHOUSE = "open_warehouse"
    TRANSFER_SLOT = "transfer_slot"
    RETURN_BASE = "return_base"


class WarehouseCleanupPhase(StrEnum):
    CONFIRMING_BASE = "confirming_base"
    WAITING_WAREHOUSE = "waiting_warehouse"
    WAITING_TRANSFER = "waiting_transfer"
    WAITING_BASE = "waiting_base"
    COMPLETED = "completed"
    STOPPED = "stopped"


class WarehouseCleanupStatus(StrEnum):
    OBSERVING = "observing"
    ACTION_READY = "action_ready"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class WarehouseCleanupPolicy:
    confirmation_frames: int = 3
    maximum_confirmation_span_ms: int = 750
    maximum_frame_age_ms: int = 250
    transition_timeout_ms: int = 8000
    maximum_action_point_drift_px: float = 8.0
    nonempty_transfer_verified: bool = False
    expected_frame_size: tuple[int, int] = (1920, 1080)

    def __post_init__(self) -> None:
        for field, value in (
            ("confirmation_frames", self.confirmation_frames),
            ("maximum_confirmation_span_ms", self.maximum_confirmation_span_ms),
            ("maximum_frame_age_ms", self.maximum_frame_age_ms),
            ("transition_timeout_ms", self.transition_timeout_ms),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} 必须是正整数")
        drift = self.maximum_action_point_drift_px
        if (
            isinstance(drift, bool)
            or not isinstance(drift, (int, float))
            or not math.isfinite(drift)
            or drift < 0
        ):
            raise ValueError("maximum_action_point_drift_px 必须是非负有限数")
        if type(self.nonempty_transfer_verified) is not bool:
            raise ValueError("nonempty_transfer_verified 必须是布尔值")
        if self.expected_frame_size != (1920, 1080):
            raise ValueError("仓库清理只允许 1920x1080")


def _valid_point(point: tuple[float, float] | None) -> bool:
    return bool(
        point is None
        or (
            isinstance(point, tuple)
            and len(point) == 2
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                for value in point
            )
        )
    )


@dataclass(frozen=True, slots=True)
class WarehouseObservation:
    frame_sequence: int
    captured_at_ns: int
    frame_size: tuple[int, int]
    scene: WarehouseScene
    accepted: bool
    safe_box_count: int | None = None
    slots: tuple[SafeSlotState, SafeSlotState] = (
        SafeSlotState.UNKNOWN,
        SafeSlotState.UNKNOWN,
    )
    open_warehouse_point: tuple[float, float] | None = None
    return_base_point: tuple[float, float] | None = None
    transfer_points: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
    ] = (None, None)

    def __post_init__(self) -> None:
        if type(self.frame_sequence) is not int or self.frame_sequence < 0:
            raise ValueError("截图序号必须是非负整数")
        if type(self.captured_at_ns) is not int or self.captured_at_ns < 0:
            raise ValueError("截图时间戳必须是非负整数")
        if (
            not isinstance(self.frame_size, tuple)
            or len(self.frame_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.frame_size)
        ):
            raise ValueError("截图尺寸必须包含正整数宽高")
        if not isinstance(self.scene, WarehouseScene):
            raise ValueError("scene 必须是 WarehouseScene")
        if type(self.accepted) is not bool:
            raise ValueError("accepted 必须是布尔值")
        if self.safe_box_count is not None and (
            type(self.safe_box_count) is not int or not 0 <= self.safe_box_count <= 2
        ):
            raise ValueError("safe_box_count 必须是 0 到 2 或 None")
        if len(self.slots) != 2 or any(not isinstance(slot, SafeSlotState) for slot in self.slots):
            raise ValueError("slots 必须包含两个 SafeSlotState")
        if len(self.transfer_points) != 2:
            raise ValueError("transfer_points 必须包含两个位置")
        points = (
            self.open_warehouse_point,
            self.return_base_point,
            *self.transfer_points,
        )
        if any(not _valid_point(point) for point in points):
            raise ValueError("动作位置必须是有限坐标或 None")
        width, height = self.frame_size
        if any(
            point is not None and not (0 <= point[0] < width and 0 <= point[1] < height)
            for point in points
        ):
            raise ValueError("动作位置必须位于截图范围内")


@dataclass(frozen=True, slots=True)
class CleanupActionIntent:
    intent_id: str
    kind: CleanupIntentKind
    position: tuple[float, float]
    expires_at_ns: int
    slot_index: int | None = None

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("清理动作 intent_id 不能为空")
        if not isinstance(self.kind, CleanupIntentKind):
            raise ValueError("清理动作 kind 无效")
        if self.position is None or not _valid_point(self.position):
            raise ValueError("清理动作位置无效")
        if type(self.expires_at_ns) is not int or self.expires_at_ns <= 0:
            raise ValueError("清理动作过期时间必须是正整数")
        if self.kind is CleanupIntentKind.TRANSFER_SLOT:
            if self.slot_index not in {0, 1}:
                raise ValueError("转移动作必须指定保险箱格子")
        elif self.slot_index is not None:
            raise ValueError("非转移动作不能指定保险箱格子")


@dataclass(frozen=True, slots=True)
class WarehouseCleanupSnapshot:
    status: WarehouseCleanupStatus
    phase: WarehouseCleanupPhase
    action: CleanupActionIntent | None
    reason: str | None
    observed_scene: WarehouseScene
    safe_box_count: int | None


class WarehouseCleanupController:
    """只在连续确认和严格后置条件成立时生成一次性动作意图。"""

    def __init__(self, policy: WarehouseCleanupPolicy) -> None:
        if not isinstance(policy, WarehouseCleanupPolicy):
            raise TypeError("policy 必须是 WarehouseCleanupPolicy")
        self._policy = policy
        self._phase = WarehouseCleanupPhase.CONFIRMING_BASE
        self._previous_sequence = -1
        self._previous_captured_at_ns = -1
        self._phase_started_at_ns: int | None = None
        self._confirmation_key: tuple[object, ...] | None = None
        self._confirmation_count = 0
        self._confirmation_started_at_ns = 0
        self._confirmation_point: tuple[float, float] | None = None
        self._transfer_baseline_count: int | None = None
        self._transfer_slot: int | None = None
        self._consumed_transfer_slots: set[int] = set()
        self._terminal: WarehouseCleanupSnapshot | None = None

    def _snapshot(
        self,
        observation: WarehouseObservation,
        *,
        status: WarehouseCleanupStatus = WarehouseCleanupStatus.OBSERVING,
        action: CleanupActionIntent | None = None,
        reason: str | None = None,
    ) -> WarehouseCleanupSnapshot:
        return WarehouseCleanupSnapshot(
            status=status,
            phase=self._phase,
            action=action,
            reason=reason,
            observed_scene=observation.scene,
            safe_box_count=observation.safe_box_count,
        )

    def _stop(
        self,
        observation: WarehouseObservation,
        reason: str,
    ) -> WarehouseCleanupSnapshot:
        self._phase = WarehouseCleanupPhase.STOPPED
        self._terminal = self._snapshot(
            observation,
            status=WarehouseCleanupStatus.STOPPED,
            reason=reason,
        )
        return self._terminal

    def stop(self, reason: str) -> WarehouseCleanupSnapshot:
        if self._terminal is not None:
            return self._terminal
        synthetic = WarehouseObservation(
            frame_sequence=max(0, self._previous_sequence + 1),
            captured_at_ns=max(0, self._previous_captured_at_ns + 1),
            frame_size=self._policy.expected_frame_size,
            scene=WarehouseScene.UNKNOWN,
            accepted=False,
        )
        return self._stop(synthetic, reason)

    def _reset_confirmation(self) -> None:
        self._confirmation_key = None
        self._confirmation_count = 0
        self._confirmation_started_at_ns = 0
        self._confirmation_point = None

    def _set_phase(self, phase: WarehouseCleanupPhase, *, now_ns: int) -> None:
        self._phase = phase
        self._phase_started_at_ns = now_ns
        self._reset_confirmation()

    def _intent(
        self,
        *,
        intent_id: str,
        kind: CleanupIntentKind,
        point: tuple[float, float],
        captured_at_ns: int,
        slot_index: int | None = None,
    ) -> CleanupActionIntent:
        return CleanupActionIntent(
            intent_id=intent_id,
            kind=kind,
            position=point,
            expires_at_ns=(captured_at_ns + self._policy.maximum_frame_age_ms * 1_000_000),
            slot_index=slot_index,
        )

    def _confirmed(
        self,
        observation: WarehouseObservation,
        *,
        key: tuple[object, ...],
        action_point: tuple[float, float] | None,
    ) -> bool:
        span_ns = self._policy.maximum_confirmation_span_ms * 1_000_000
        drift_limit = self._policy.maximum_action_point_drift_px
        point_drifted = bool(
            action_point is not None
            and self._confirmation_point is not None
            and math.dist(action_point, self._confirmation_point) > drift_limit
        )
        if (
            self._confirmation_key != key
            or observation.captured_at_ns - self._confirmation_started_at_ns > span_ns
            or point_drifted
        ):
            self._confirmation_key = key
            self._confirmation_count = 1
            self._confirmation_started_at_ns = observation.captured_at_ns
            self._confirmation_point = action_point
        else:
            self._confirmation_count += 1
        return self._confirmation_count >= self._policy.confirmation_frames

    def _validate_observation(
        self,
        observation: WarehouseObservation,
        *,
        now_ns: int,
    ) -> str | None:
        if type(now_ns) is not int or now_ns < 0:
            return "清理时钟必须是非负整数纳秒"
        if observation.frame_sequence <= self._previous_sequence:
            return "截图序号必须严格递增"
        if observation.captured_at_ns <= self._previous_captured_at_ns:
            return "截图时间戳必须严格递增"
        self._previous_sequence = observation.frame_sequence
        self._previous_captured_at_ns = observation.captured_at_ns
        frame_age_ns = now_ns - observation.captured_at_ns
        if frame_age_ns < 0 or frame_age_ns > self._policy.maximum_frame_age_ms * 1_000_000:
            return "截图已经过期或来自未来"
        if observation.frame_size != self._policy.expected_frame_size:
            return "截图分辨率不是 1920x1080"
        if not observation.accepted or observation.scene is WarehouseScene.UNKNOWN:
            return "仓库页面观察未知或存在歧义"
        if self._phase_started_at_ns is None:
            self._phase_started_at_ns = now_ns
        elif now_ns - self._phase_started_at_ns >= self._policy.transition_timeout_ms * 1_000_000:
            return "仓库清理页面转换超时"
        return None

    @staticmethod
    def _inventory_error(observation: WarehouseObservation) -> str | None:
        if observation.safe_box_count is None:
            return "未确认安全箱计数"
        if any(slot is SafeSlotState.UNKNOWN for slot in observation.slots):
            return "安全箱格子状态未知"
        occupied = sum(slot is SafeSlotState.OCCUPIED for slot in observation.slots)
        if occupied != observation.safe_box_count:
            return "安全箱计数与格子占用不一致"
        return None

    def _inventory_intent(
        self,
        observation: WarehouseObservation,
        *,
        now_ns: int,
    ) -> CleanupActionIntent | WarehouseCleanupSnapshot:
        assert observation.safe_box_count is not None
        if observation.safe_box_count == 0:
            if observation.return_base_point is None:
                return self._stop(observation, "已确认空保险箱，但缺少返回 BASE 动作锚点")
            action = self._intent(
                intent_id="return-base",
                kind=CleanupIntentKind.RETURN_BASE,
                point=observation.return_base_point,
                captured_at_ns=observation.captured_at_ns,
            )
            self._set_phase(WarehouseCleanupPhase.WAITING_BASE, now_ns=now_ns)
            return action
        if not self._policy.nonempty_transfer_verified:
            return self._stop(observation, "非空保险箱转移动作尚未验证")
        slot_index = next(
            index
            for index, state in enumerate(observation.slots)
            if state is SafeSlotState.OCCUPIED
        )
        if slot_index in self._consumed_transfer_slots:
            return self._stop(observation, "已消费的保险箱格子再次显示占用")
        point = observation.transfer_points[slot_index]
        if point is None:
            return self._stop(observation, "已确认非空格子，但缺少独立转移动作锚点")
        self._consumed_transfer_slots.add(slot_index)
        self._transfer_baseline_count = observation.safe_box_count
        self._transfer_slot = slot_index
        action = self._intent(
            intent_id=f"transfer-safe-slot-{slot_index}",
            kind=CleanupIntentKind.TRANSFER_SLOT,
            point=point,
            captured_at_ns=observation.captured_at_ns,
            slot_index=slot_index,
        )
        self._set_phase(WarehouseCleanupPhase.WAITING_TRANSFER, now_ns=now_ns)
        return action

    def _warehouse_step(
        self,
        observation: WarehouseObservation,
        *,
        now_ns: int,
    ) -> WarehouseCleanupSnapshot:
        error = self._inventory_error(observation)
        if error is not None:
            return self._stop(observation, error)
        assert observation.safe_box_count is not None
        if observation.safe_box_count > 0 and not self._policy.nonempty_transfer_verified:
            return self._stop(observation, "非空保险箱转移动作尚未验证")
        key = (
            observation.scene,
            observation.safe_box_count,
            observation.slots,
        )
        relevant_point = (
            observation.return_base_point
            if observation.safe_box_count == 0
            else observation.transfer_points[
                next(
                    index
                    for index, state in enumerate(observation.slots)
                    if state is SafeSlotState.OCCUPIED
                )
            ]
        )
        if relevant_point is None:
            return self._stop(observation, "需要清理动作，但独立动作锚点缺失")
        if not self._confirmed(observation, key=key, action_point=relevant_point):
            return self._snapshot(observation)
        intent = self._inventory_intent(observation, now_ns=now_ns)
        if isinstance(intent, WarehouseCleanupSnapshot):
            return intent
        return self._snapshot(
            observation,
            status=WarehouseCleanupStatus.ACTION_READY,
            action=intent,
        )

    def _transfer_step(
        self,
        observation: WarehouseObservation,
        *,
        now_ns: int,
    ) -> WarehouseCleanupSnapshot:
        error = self._inventory_error(observation)
        if error is not None:
            return self._stop(observation, error)
        assert self._transfer_baseline_count is not None
        assert self._transfer_slot is not None
        assert observation.safe_box_count is not None
        if (
            observation.safe_box_count == self._transfer_baseline_count
            and observation.slots[self._transfer_slot] is SafeSlotState.OCCUPIED
        ):
            self._reset_confirmation()
            return self._snapshot(observation)
        if (
            observation.safe_box_count != self._transfer_baseline_count - 1
            or observation.slots[self._transfer_slot] is not SafeSlotState.EMPTY
        ):
            return self._stop(observation, "转移后的计数或目标格变化不符合预期")
        key = (
            observation.safe_box_count,
            observation.slots,
        )
        relevant_point = (
            observation.return_base_point
            if observation.safe_box_count == 0
            else observation.transfer_points[
                next(
                    index
                    for index, state in enumerate(observation.slots)
                    if state is SafeSlotState.OCCUPIED
                )
            ]
        )
        if relevant_point is None:
            return self._stop(observation, "转移后下一动作的独立锚点缺失")
        if not self._confirmed(
            observation,
            key=key,
            action_point=relevant_point,
        ):
            return self._snapshot(observation)
        intent = self._inventory_intent(observation, now_ns=now_ns)
        if isinstance(intent, WarehouseCleanupSnapshot):
            return intent
        return self._snapshot(
            observation,
            status=WarehouseCleanupStatus.ACTION_READY,
            action=intent,
        )

    def step(
        self,
        observation: WarehouseObservation,
        *,
        now_ns: int,
    ) -> WarehouseCleanupSnapshot:
        if self._terminal is not None:
            return self._terminal
        if not isinstance(observation, WarehouseObservation):
            raise TypeError("observation 必须是 WarehouseObservation")
        error = self._validate_observation(observation, now_ns=now_ns)
        if error is not None:
            return self._stop(observation, error)
        if (
            self._phase is WarehouseCleanupPhase.WAITING_WAREHOUSE
            and observation.scene is WarehouseScene.BASE
        ):
            return self._snapshot(observation)
        if (
            self._phase is WarehouseCleanupPhase.WAITING_BASE
            and observation.scene is WarehouseScene.WAREHOUSE
        ):
            inventory_error = self._inventory_error(observation)
            if inventory_error is not None:
                return self._stop(observation, inventory_error)
            if observation.safe_box_count != 0:
                return self._stop(observation, "等待返回 BASE 时保险箱不再为空")
            return self._snapshot(observation)
        expected_scene = (
            WarehouseScene.BASE
            if self._phase
            in {
                WarehouseCleanupPhase.CONFIRMING_BASE,
                WarehouseCleanupPhase.WAITING_BASE,
            }
            else WarehouseScene.WAREHOUSE
        )
        if observation.scene is not expected_scene:
            return self._stop(observation, "观察页面与仓库清理阶段不一致")
        if self._phase is WarehouseCleanupPhase.CONFIRMING_BASE:
            point = observation.open_warehouse_point
            if point is None:
                return self._stop(observation, "BASE 已确认但缺少打开仓库动作锚点")
            if not self._confirmed(
                observation,
                key=(observation.scene,),
                action_point=point,
            ):
                return self._snapshot(observation)
            action = self._intent(
                intent_id="open-warehouse",
                kind=CleanupIntentKind.OPEN_WAREHOUSE,
                point=point,
                captured_at_ns=observation.captured_at_ns,
            )
            self._set_phase(WarehouseCleanupPhase.WAITING_WAREHOUSE, now_ns=now_ns)
            return self._snapshot(
                observation,
                status=WarehouseCleanupStatus.ACTION_READY,
                action=action,
            )
        if self._phase is WarehouseCleanupPhase.WAITING_WAREHOUSE:
            return self._warehouse_step(observation, now_ns=now_ns)
        if self._phase is WarehouseCleanupPhase.WAITING_TRANSFER:
            return self._transfer_step(observation, now_ns=now_ns)
        if self._phase is WarehouseCleanupPhase.WAITING_BASE:
            if not self._confirmed(
                observation,
                key=(observation.scene,),
                action_point=None,
            ):
                return self._snapshot(observation)
            self._phase = WarehouseCleanupPhase.COMPLETED
            self._terminal = self._snapshot(
                observation,
                status=WarehouseCleanupStatus.COMPLETED,
            )
            return self._terminal
        return self._stop(observation, "仓库清理进入未知阶段")
