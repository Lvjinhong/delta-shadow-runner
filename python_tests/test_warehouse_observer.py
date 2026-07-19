from types import SimpleNamespace

import numpy as np
import pytest

from delta_vision.frames import CapturedFrame
from delta_vision.menu_automation import (
    MenuActionKind,
    MenuScene,
    MenuTransition,
    SceneDecisionReason,
    SceneObservation,
)
from delta_vision.warehouse_cleanup import (
    WAREHOUSE_BASE_TEMPLATE_ID,
    WAREHOUSE_EMPTY_EVIDENCE_IDS,
    WAREHOUSE_EMPTY_TEMPLATE_ID,
    MenuProfileWarehouseObserver,
    SafeSlotState,
    WarehouseScene,
)


class FakeSceneObserver:
    def __init__(self, observation: SceneObservation) -> None:
        self.observation = observation
        self.frames: list[CapturedFrame] = []

    def observe(self, frame: CapturedFrame) -> SceneObservation:
        self.frames.append(frame)
        return self.observation


def _frame(*, size: tuple[int, int] = (1920, 1080)) -> CapturedFrame:
    width, height = size
    return CapturedFrame(
        sequence=7,
        captured_at_ns=123_000_000,
        image=np.zeros((height, width, 3), dtype=np.uint8),
        source="test",
    )


def _scene_observation(
    *,
    scene: MenuScene,
    accepted: bool = True,
    action_point: tuple[float, float] | None = (325.0, 55.0),
    template_id: str = "base-home",
) -> SceneObservation:
    return SceneObservation(
        frame_sequence=7,
        captured_at_ns=123_000_000,
        scene=scene if accepted else MenuScene.UNKNOWN,
        candidate_scene=scene,
        confidence=0.98,
        runner_up_confidence=0.2,
        accepted=accepted,
        reason=(
            SceneDecisionReason.ACCEPTED
            if accepted
            else SceneDecisionReason.BELOW_THRESHOLD
        ),
        action_accepted=accepted and action_point is not None,
        action_point=action_point if accepted else None,
        page_point=(100.0, 100.0) if accepted else None,
        page_template_id=template_id,
    )


def _provenance(*, template_id: str, scene: MenuScene):
    evidence_ids = (
        tuple(sorted(WAREHOUSE_EMPTY_EVIDENCE_IDS))
        if template_id == WAREHOUSE_EMPTY_TEMPLATE_ID
        else ()
    )
    return SimpleNamespace(
        template_id=template_id,
        scene=scene,
        page_content_sha256=f"page-{template_id}",
        action_content_sha256=f"action-{template_id}",
        action_sha256=f"sha-{template_id}",
        evidence_ids=evidence_ids,
        evidence_sha256=tuple(f"sha-{item}" for item in evidence_ids),
        evidence_content_sha256=tuple(
            f"content-{item}" for item in evidence_ids
        ),
    )


def _profile(
    observation: SceneObservation,
    *,
    frame_size=(1920, 1080),
    provenance=None,
    transitions=None,
):
    default_transitions = (
        MenuTransition(
            source=MenuScene.BASE,
            target=MenuScene.WAREHOUSE,
            action_kind=MenuActionKind.CLICK,
        ),
        MenuTransition(
            source=MenuScene.WAREHOUSE,
            target=MenuScene.BASE,
            action_kind=MenuActionKind.CLICK,
        ),
    )
    return SimpleNamespace(
        frame_size=frame_size,
        observer=FakeSceneObserver(observation),
        template_provenance=(
            _provenance(template_id=WAREHOUSE_BASE_TEMPLATE_ID, scene=MenuScene.BASE),
            _provenance(
                template_id=WAREHOUSE_EMPTY_TEMPLATE_ID,
                scene=MenuScene.WAREHOUSE,
            ),
        )
        if provenance is None
        else provenance,
        transitions=default_transitions if transitions is None else transitions,
        stop_scenes=frozenset(),
    )


def test_base_scene_maps_only_independent_warehouse_action() -> None:
    source = _scene_observation(
        scene=MenuScene.BASE,
        action_point=(325.0, 55.0),
        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
    )
    profile = _profile(source)
    observer = MenuProfileWarehouseObserver(menu_profile=profile)

    result = observer.observe(_frame())

    assert result.scene is WarehouseScene.BASE
    assert result.accepted is True
    assert result.open_warehouse_point == (325.0, 55.0)
    assert result.return_base_point is None
    assert result.safe_box_count is None
    assert result.slots == (SafeSlotState.UNKNOWN, SafeSlotState.UNKNOWN)


def test_positive_empty_template_maps_count_slots_and_return_action() -> None:
    source = _scene_observation(
        scene=MenuScene.WAREHOUSE,
        action_point=(190.0, 55.0),
        template_id=WAREHOUSE_EMPTY_TEMPLATE_ID,
    )
    observer = MenuProfileWarehouseObserver(menu_profile=_profile(source))

    result = observer.observe(_frame())

    assert result.scene is WarehouseScene.WAREHOUSE
    assert result.accepted is True
    assert result.safe_box_count == 0
    assert result.slots == (SafeSlotState.EMPTY, SafeSlotState.EMPTY)
    assert result.return_base_point == (190.0, 55.0)
    assert result.open_warehouse_point is None


@pytest.mark.parametrize(
    "source",
    [
        _scene_observation(scene=MenuScene.BASE, accepted=False),
        _scene_observation(
            scene=MenuScene.WAREHOUSE,
            action_point=None,
            template_id=WAREHOUSE_EMPTY_TEMPLATE_ID,
        ),
        _scene_observation(
            scene=MenuScene.WAREHOUSE,
            template_id="warehouse-unverified-nonempty",
        ),
        _scene_observation(scene=MenuScene.IN_MATCH),
    ],
)
def test_unknown_missing_action_nonempty_or_unrelated_scene_fails_closed(
    source: SceneObservation,
) -> None:
    observer = MenuProfileWarehouseObserver(menu_profile=_profile(source))

    result = observer.observe(_frame())

    assert result.scene is WarehouseScene.UNKNOWN
    assert result.accepted is False
    assert result.safe_box_count is None
    assert result.slots == (SafeSlotState.UNKNOWN, SafeSlotState.UNKNOWN)
    assert result.open_warehouse_point is None
    assert result.return_base_point is None


def test_profile_requires_1080p_and_declared_empty_warehouse_template() -> None:
    source = _scene_observation(scene=MenuScene.BASE)

    with pytest.raises(ValueError, match="1920x1080"):
        MenuProfileWarehouseObserver(
            menu_profile=_profile(source, frame_size=(2560, 1440)),
        )

    with pytest.raises(ValueError, match="空保险箱模板"):
        MenuProfileWarehouseObserver(
            menu_profile=_profile(
                source,
                provenance=(
                    _provenance(
                        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
                        scene=MenuScene.BASE,
                    ),
                ),
            )
        )


def test_observer_preserves_frame_identity_and_actual_size() -> None:
    source = _scene_observation(scene=MenuScene.BASE)
    observer = MenuProfileWarehouseObserver(menu_profile=_profile(source))

    result = observer.observe(_frame(size=(1280, 720)))

    assert result.frame_sequence == 7
    assert result.captured_at_ns == 123_000_000
    assert result.frame_size == (1280, 720)
    assert result.scene is WarehouseScene.UNKNOWN
    assert result.accepted is False


@pytest.mark.parametrize(
    "profile_override, error",
    [
        (
            {
                "provenance": (
                    _provenance(
                        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
                        scene=MenuScene.BASE,
                    ),
                    _provenance(
                        template_id=WAREHOUSE_EMPTY_TEMPLATE_ID,
                        scene=MenuScene.WAREHOUSE,
                    ),
                    _provenance(template_id="extra-base", scene=MenuScene.BASE),
                )
            },
            "恰好",
        ),
        (
            {
                "transitions": (
                    MenuTransition(
                        source=MenuScene.BASE,
                        target=MenuScene.WAREHOUSE,
                        action_kind=MenuActionKind.CLICK,
                    ),
                )
            },
            "转换",
        ),
    ],
)
def test_profile_rejects_extra_templates_or_wrong_transition_topology(
    profile_override: dict[str, object],
    error: str,
) -> None:
    source = _scene_observation(
        scene=MenuScene.BASE,
        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
    )

    with pytest.raises(ValueError, match=error):
        MenuProfileWarehouseObserver(
            menu_profile=_profile(source, **profile_override),
        )


@pytest.mark.parametrize("missing", ["action_sha256", "action_content_sha256"])
def test_profile_rejects_missing_or_shared_action_provenance(missing: str) -> None:
    source = _scene_observation(
        scene=MenuScene.BASE,
        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
    )
    base = _provenance(
        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
        scene=MenuScene.BASE,
    )
    setattr(base, missing, None)

    with pytest.raises(ValueError, match="独立动作"):
        MenuProfileWarehouseObserver(
            menu_profile=_profile(
                source,
                provenance=(
                    base,
                    _provenance(
                        template_id=WAREHOUSE_EMPTY_TEMPLATE_ID,
                        scene=MenuScene.WAREHOUSE,
                    ),
                ),
            )
        )


def test_profile_rejects_missing_count_or_slot_evidence() -> None:
    source = _scene_observation(
        scene=MenuScene.BASE,
        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
    )
    empty = _provenance(
        template_id=WAREHOUSE_EMPTY_TEMPLATE_ID,
        scene=MenuScene.WAREHOUSE,
    )
    empty.evidence_ids = empty.evidence_ids[:-1]
    empty.evidence_sha256 = empty.evidence_sha256[:-1]
    empty.evidence_content_sha256 = empty.evidence_content_sha256[:-1]

    with pytest.raises(ValueError, match="计数和两个独立空格证据"):
        MenuProfileWarehouseObserver(
            menu_profile=_profile(
                source,
                provenance=(
                    _provenance(
                        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
                        scene=MenuScene.BASE,
                    ),
                    empty,
                ),
            )
        )


def test_observer_is_called_once_and_controller_is_never_created() -> None:
    source = _scene_observation(
        scene=MenuScene.BASE,
        template_id=WAREHOUSE_BASE_TEMPLATE_ID,
    )
    profile = _profile(source)
    profile.create_controller = lambda: pytest.fail("适配器不能创建菜单控制器")
    observer = MenuProfileWarehouseObserver(menu_profile=profile)

    observer.observe(_frame())

    assert len(profile.observer.frames) == 1
