import numpy as np
import pytest

from delta_vision.config import CaptureRegion
from delta_vision.frames import CapturedFrame
from delta_vision.menu_automation import (
    MenuActionKind,
    MenuControllerStatus,
    MenuScene,
    MenuSceneTemplate,
    MenuTransition,
    SceneDecisionReason,
    SceneObservation,
    TemplateMenuSceneObserver,
    VisualMenuController,
)
from delta_vision.template_matching import (
    MatchDecisionPolicy,
    TemplateAnchorDetector,
)

FRAME_SIZE = (2560, 1440)
ACTION_REGION = CaptureRegion(left=700, top=500, width=300, height=300)


def _pattern(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        256,
        size=(20, 28, 3),
        dtype=np.uint8,
    )


def _detector(label: str, template: np.ndarray) -> TemplateAnchorDetector:
    return TemplateAnchorDetector(
        label=label,
        template=template,
        search_roi=CaptureRegion(left=0, top=0, width=2560, height=1440),
        scales=(1.0,),
        policy=MatchDecisionPolicy(score_threshold=0.9, minimum_margin=0.08),
        nms_radius_px=32,
    )


def _observer(
    *,
    duplicate_scene_template: bool = False,
) -> tuple[
    TemplateMenuSceneObserver,
    dict[MenuScene, tuple[np.ndarray, np.ndarray | None]],
]:
    patterns: dict[MenuScene, tuple[np.ndarray, np.ndarray | None]] = {
        MenuScene.LOBBY: (_pattern(11), _pattern(111)),
        MenuScene.STRATEGY_BOARD: (_pattern(12), _pattern(112)),
        MenuScene.ZERO_DAM_READY: (_pattern(13), _pattern(113)),
        MenuScene.IN_MATCH: (_pattern(14), None),
        MenuScene.DEATH_SUMMARY: (_pattern(15), None),
    }
    strategy_pattern = (
        patterns[MenuScene.LOBBY][0]
        if duplicate_scene_template
        else patterns[MenuScene.STRATEGY_BOARD][0]
    )
    observer = TemplateMenuSceneObserver(
        templates=(
            MenuSceneTemplate(
                "lobby-preparation",
                MenuScene.LOBBY,
                _detector("lobby-page", patterns[MenuScene.LOBBY][0]),
                _detector("lobby-preparation", patterns[MenuScene.LOBBY][1]),
                ACTION_REGION,
            ),
            MenuSceneTemplate(
                "strategy-zero-dam",
                MenuScene.STRATEGY_BOARD,
                _detector("strategy-page", strategy_pattern),
                _detector(
                    "strategy-zero-dam",
                    patterns[MenuScene.STRATEGY_BOARD][1],
                ),
                ACTION_REGION,
            ),
            MenuSceneTemplate(
                "zero-dam-depart",
                MenuScene.ZERO_DAM_READY,
                _detector("zero-dam-page", patterns[MenuScene.ZERO_DAM_READY][0]),
                _detector(
                    "zero-dam-depart",
                    patterns[MenuScene.ZERO_DAM_READY][1],
                ),
                ACTION_REGION,
            ),
            MenuSceneTemplate(
                "in-match-hud",
                MenuScene.IN_MATCH,
                _detector("in-match-hud", patterns[MenuScene.IN_MATCH][0]),
                None,
                None,
            ),
            MenuSceneTemplate(
                "death-summary",
                MenuScene.DEATH_SUMMARY,
                _detector("death-summary", patterns[MenuScene.DEATH_SUMMARY][0]),
                None,
                None,
            ),
        ),
        expected_frame_size=FRAME_SIZE,
        minimum_scene_margin=0.08,
    )
    return observer, patterns


def _frame(
    sequence: int,
    patterns: tuple[np.ndarray, np.ndarray | None] | None,
    *,
    captured_at_ns: int | None = None,
    position: tuple[int, int] = (800, 600),
    page_position: tuple[int, int] = (100, 100),
    size: tuple[int, int] = FRAME_SIZE,
) -> CapturedFrame:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if patterns is not None:
        page_pattern, action_pattern = patterns
        page_x, page_y = page_position
        page_height, page_width = page_pattern.shape[:2]
        image[
            page_y : page_y + page_height,
            page_x : page_x + page_width,
        ] = page_pattern
    else:
        action_pattern = None
    if action_pattern is not None:
        x, y = position
        action_height, action_width = action_pattern.shape[:2]
        image[y : y + action_height, x : x + action_width] = action_pattern
    image.setflags(write=False)
    return CapturedFrame(
        sequence=sequence,
        captured_at_ns=(
            sequence * 100_000_000 if captured_at_ns is None else captured_at_ns
        ),
        image=image,
        source="menu-fixture",
    )


def _controller() -> VisualMenuController:
    return VisualMenuController(
        transitions=(
            MenuTransition(
                source=MenuScene.LOBBY,
                target=MenuScene.STRATEGY_BOARD,
                action_kind=MenuActionKind.CLICK,
            ),
            MenuTransition(
                source=MenuScene.STRATEGY_BOARD,
                target=MenuScene.ZERO_DAM_READY,
                action_kind=MenuActionKind.CLICK,
            ),
            MenuTransition(
                source=MenuScene.ZERO_DAM_READY,
                target=MenuScene.IN_MATCH,
                action_kind=MenuActionKind.CLICK,
            ),
        ),
        confirmation_frames=3,
        maximum_confirmation_span_ms=750,
        maximum_action_point_drift_px=12,
        maximum_page_point_drift_px=12,
        maximum_frame_age_ms=250,
        transition_timeout_ms=8_000,
        stop_scenes=frozenset({MenuScene.DEATH_SUMMARY}),
    )


def _step(
    controller: VisualMenuController,
    observation,
    *,
    now_ns: int | None = None,
):
    return controller.step(
        observation,
        now_ns=observation.captured_at_ns if now_ns is None else now_ns,
    )


def test_template_scene_observer_recognizes_scene_and_click_point() -> None:
    observer, patterns = _observer()

    observation = observer.observe(_frame(1, patterns[MenuScene.LOBBY]))

    assert observation.scene is MenuScene.LOBBY
    assert observation.accepted is True
    assert observation.reason is SceneDecisionReason.ACCEPTED
    assert observation.confidence >= 0.99
    assert observation.runner_up_confidence < 0.9
    assert observation.action_point == (814.0, 610.0)


def test_template_scene_observer_requires_every_evidence_anchor() -> None:
    page = _pattern(201)
    action = _pattern(202)
    evidence = _pattern(203)
    observer = TemplateMenuSceneObserver(
        templates=(
            MenuSceneTemplate(
                template_id="warehouse-empty",
                scene=MenuScene.WAREHOUSE,
                page_detector=_detector("warehouse-page", page),
                action_detector=_detector("warehouse-return", action),
                action_region=ACTION_REGION,
                evidence_detectors=(_detector("safe-box-zero", evidence),),
            ),
        ),
        expected_frame_size=FRAME_SIZE,
        minimum_scene_margin=0.08,
    )
    complete_source = _frame(1, (page, action))
    complete_image = np.array(complete_source.image, copy=True)
    complete_image[300:320, 400:428] = evidence
    complete_image.setflags(write=False)
    complete = CapturedFrame(
        sequence=complete_source.sequence,
        captured_at_ns=complete_source.captured_at_ns,
        image=complete_image,
        source=complete_source.source,
    )
    missing = _frame(2, (page, action))

    accepted = observer.observe(complete)
    rejected = observer.observe(missing)

    assert accepted.scene is MenuScene.WAREHOUSE
    assert accepted.accepted is True
    assert accepted.action_accepted is True
    assert rejected.scene is MenuScene.UNKNOWN
    assert rejected.accepted is False
    assert rejected.reason is SceneDecisionReason.BELOW_THRESHOLD


def test_template_scene_observer_rejects_cross_scene_ambiguity() -> None:
    observer, patterns = _observer(duplicate_scene_template=True)

    observation = observer.observe(_frame(1, patterns[MenuScene.LOBBY]))

    assert observation.scene is MenuScene.UNKNOWN
    assert observation.accepted is False
    assert observation.reason is SceneDecisionReason.AMBIGUOUS
    assert observation.candidate_scene is MenuScene.LOBBY
    assert observation.confidence >= 0.99
    assert observation.runner_up_confidence >= 0.99
    assert observation.action_point is None


def test_template_scene_observer_rejects_low_confidence_and_wrong_resolution() -> None:
    observer, _ = _observer()

    blank = observer.observe(_frame(1, None))
    wrong_size = observer.observe(_frame(2, None, size=(1920, 1080)))

    assert blank.scene is MenuScene.UNKNOWN
    assert blank.accepted is False
    assert blank.reason is SceneDecisionReason.BELOW_THRESHOLD
    assert wrong_size.scene is MenuScene.UNKNOWN
    assert wrong_size.accepted is False
    assert wrong_size.reason is SceneDecisionReason.FRAME_SIZE_MISMATCH


def test_template_scene_observer_requires_independent_action_anchor() -> None:
    observer, patterns = _observer()
    page_pattern, _ = patterns[MenuScene.LOBBY]

    observation = observer.observe(_frame(1, (page_pattern, None)))

    assert observation.scene is MenuScene.LOBBY
    assert observation.accepted is True
    assert observation.reason is SceneDecisionReason.ACCEPTED
    assert observation.action_point is None
    assert observation.action_accepted is False


def test_controller_requires_three_stable_frames_before_clicking() -> None:
    observer, patterns = _observer()
    controller = _controller()

    first = _step(controller, observer.observe(_frame(1, patterns[MenuScene.LOBBY])))
    second = _step(controller, observer.observe(_frame(2, patterns[MenuScene.LOBBY])))
    third = _step(controller, observer.observe(_frame(3, patterns[MenuScene.LOBBY])))

    assert first.action is None
    assert second.action is None
    assert third.status is MenuControllerStatus.WAITING_FOR_TRANSITION
    assert third.action is not None
    assert third.action.kind is MenuActionKind.CLICK
    assert third.action.position == (814, 610)
    assert third.action.expires_at_ns == 550_000_000


def test_controller_resets_confirmation_on_unknown_or_click_point_drift() -> None:
    observer, patterns = _observer()
    controller = _controller()
    lobby = patterns[MenuScene.LOBBY]

    snapshots = [
        _step(controller, observer.observe(_frame(1, lobby, position=(800, 600)))),
        _step(controller, observer.observe(_frame(2, lobby, position=(805, 600)))),
        _step(controller, observer.observe(_frame(3, lobby, position=(830, 600)))),
        _step(controller, observer.observe(_frame(4, None))),
        _step(controller, observer.observe(_frame(5, lobby, position=(800, 600)))),
        _step(controller, observer.observe(_frame(6, lobby, position=(801, 600)))),
        _step(controller, observer.observe(_frame(7, lobby, position=(802, 600)))),
    ]

    assert all(snapshot.action is None for snapshot in snapshots[:-1])
    assert snapshots[-1].action is not None
    assert snapshots[-1].action.position == (815, 610)


def test_controller_restarts_confirmation_when_three_frames_are_too_old() -> None:
    observer, patterns = _observer()
    controller = _controller()
    lobby = patterns[MenuScene.LOBBY]

    first = _step(
        controller, observer.observe(_frame(1, lobby, captured_at_ns=100_000_000))
    )
    second = _step(
        controller, observer.observe(_frame(2, lobby, captured_at_ns=200_000_000))
    )
    stale_third = _step(
        controller, observer.observe(_frame(3, lobby, captured_at_ns=900_000_001))
    )
    fourth = _step(
        controller, observer.observe(_frame(4, lobby, captured_at_ns=950_000_000))
    )
    fifth = _step(
        controller, observer.observe(_frame(5, lobby, captured_at_ns=1_000_000_000))
    )

    assert first.action is None
    assert second.action is None
    assert stale_third.action is None
    assert fourth.action is None
    assert fifth.action is not None


def test_controller_rejects_duplicate_or_out_of_order_frames_without_action() -> None:
    observer, patterns = _observer()
    controller = _controller()
    lobby = patterns[MenuScene.LOBBY]

    _step(controller, observer.observe(_frame(1, lobby)))
    duplicate = _step(controller, observer.observe(_frame(1, lobby)))
    after_duplicate = _step(controller, observer.observe(_frame(2, lobby)))

    assert duplicate.status is MenuControllerStatus.STOPPED
    assert duplicate.action is None
    assert duplicate.reason == "截图序号或时间戳没有严格递增"
    assert after_duplicate.status is MenuControllerStatus.STOPPED
    assert after_duplicate.action is None


def test_controller_latches_action_until_target_scene_is_confirmed() -> None:
    observer, patterns = _observer()
    controller = _controller()
    lobby = patterns[MenuScene.LOBBY]
    board = patterns[MenuScene.STRATEGY_BOARD]

    for sequence in range(1, 4):
        clicked = _step(controller, observer.observe(_frame(sequence, lobby)))
    assert clicked.action is not None

    repeated_source = [
        _step(controller, observer.observe(_frame(sequence, lobby)))
        for sequence in range(4, 7)
    ]
    target = [
        _step(controller, observer.observe(_frame(sequence, board)))
        for sequence in range(7, 10)
    ]

    assert all(snapshot.action is None for snapshot in repeated_source)
    assert all(snapshot.action is None for snapshot in target)
    assert target[-1].status is MenuControllerStatus.OBSERVING
    assert target[-1].transition_index == 1


def test_controller_requires_fresh_confirmation_for_every_transition() -> None:
    observer, patterns = _observer()
    controller = _controller()
    sequence = 0

    for source, target in (
        (MenuScene.LOBBY, MenuScene.STRATEGY_BOARD),
        (MenuScene.STRATEGY_BOARD, MenuScene.ZERO_DAM_READY),
        (MenuScene.ZERO_DAM_READY, MenuScene.IN_MATCH),
    ):
        actions = []
        for _ in range(3):
            sequence += 1
            snapshot = _step(
                controller, observer.observe(_frame(sequence, patterns[source]))
            )
            actions.append(snapshot.action)
        assert actions[:2] == [None, None]
        assert actions[2] is not None

        for _ in range(3):
            sequence += 1
            snapshot = _step(
                controller, observer.observe(_frame(sequence, patterns[target]))
            )
            assert snapshot.action is None

    assert snapshot.status is MenuControllerStatus.COMPLETED
    assert snapshot.transition_index == 3


def test_controller_stops_after_transition_timeout_without_retry() -> None:
    observer, patterns = _observer()
    controller = _controller()
    lobby = patterns[MenuScene.LOBBY]

    for sequence in range(1, 4):
        clicked = _step(controller, observer.observe(_frame(sequence, lobby)))
    assert clicked.action is not None

    timeout = _step(
        controller,
        observer.observe(
            _frame(
                4,
                lobby,
                captured_at_ns=clicked.observed_at_ns + 8_000_000_001,
            )
        ),
    )
    later = _step(
        controller,
        observer.observe(_frame(5, lobby, captured_at_ns=9_000_000_000)),
    )

    assert timeout.status is MenuControllerStatus.STOPPED
    assert timeout.action is None
    assert timeout.reason == "等待目标页面超时"
    assert later.status is MenuControllerStatus.STOPPED
    assert later.action is None


def test_controller_can_be_stopped_explicitly_and_remains_terminal() -> None:
    controller = _controller()

    stopped = controller.stop("会话运行超时")
    later = controller.stop("不应覆盖原始原因")

    assert stopped.status is MenuControllerStatus.STOPPED
    assert stopped.reason == "会话运行超时"
    assert later == stopped


def test_controller_stops_on_confirmed_death_summary_without_action() -> None:
    observer, patterns = _observer()
    controller = _controller()
    death = patterns[MenuScene.DEATH_SUMMARY]

    snapshots = [
        _step(controller, observer.observe(_frame(sequence, death)))
        for sequence in range(1, 4)
    ]

    assert all(snapshot.action is None for snapshot in snapshots)
    assert snapshots[-1].status is MenuControllerStatus.STOPPED
    assert snapshots[-1].reason == "检测到停止页面: death_summary"


def test_key_transition_does_not_require_a_click_point() -> None:
    observer, patterns = _observer()
    controller = VisualMenuController(
        transitions=(
            MenuTransition(
                source=MenuScene.DEATH_SUMMARY,
                target=MenuScene.LOBBY,
                action_kind=MenuActionKind.KEY,
                key="space",
            ),
        ),
        confirmation_frames=3,
        maximum_confirmation_span_ms=750,
        maximum_action_point_drift_px=12,
        maximum_page_point_drift_px=12,
        maximum_frame_age_ms=250,
        transition_timeout_ms=8_000,
        stop_scenes=frozenset(),
    )

    for sequence in range(1, 4):
        snapshot = _step(
            controller,
            observer.observe(_frame(sequence, patterns[MenuScene.DEATH_SUMMARY])),
        )

    assert snapshot.action is not None
    assert snapshot.action.kind is MenuActionKind.KEY
    assert snapshot.action.key == "space"
    assert snapshot.action.position is None


def test_controller_rejects_frame_older_than_live_age_budget() -> None:
    observer, patterns = _observer()
    controller = _controller()
    observation = observer.observe(
        _frame(1, patterns[MenuScene.LOBBY], captured_at_ns=1_000_000_000)
    )

    snapshot = controller.step(observation, now_ns=1_250_000_001)

    assert snapshot.status is MenuControllerStatus.STOPPED
    assert snapshot.action is None
    assert snapshot.reason == "截图超过最大允许帧龄"


def test_menu_transition_rejects_non_enum_action_kind() -> None:
    try:
        MenuTransition(
            source=MenuScene.LOBBY,
            target=MenuScene.STRATEGY_BOARD,
            action_kind="click",  # type: ignore[arg-type]
        )
    except ValueError as error:
        assert "action_kind" in str(error)
    else:
        raise AssertionError("裸字符串 action_kind 必须被拒绝")


def test_controller_requires_same_page_template_for_all_confirmations() -> None:
    observer, patterns = _observer()
    controller = _controller()
    observations = [
        observer.observe(_frame(sequence, patterns[MenuScene.LOBBY]))
        for sequence in range(1, 4)
    ]
    alternate = observations[1]
    alternate = type(alternate)(
        **{
            field: getattr(alternate, field)
            for field in alternate.__dataclass_fields__
            if field != "page_template_id"
        },
        page_template_id="alternate-lobby-template",
    )

    first = controller.step(observations[0], now_ns=observations[0].captured_at_ns)
    second = controller.step(alternate, now_ns=alternate.captured_at_ns)
    third = controller.step(observations[2], now_ns=observations[2].captured_at_ns)

    assert first.action is None
    assert second.action is None
    assert third.action is None


def test_controller_rejects_unstable_page_anchor_position() -> None:
    observer, patterns = _observer()
    controller = _controller()
    lobby = patterns[MenuScene.LOBBY]
    snapshots = []

    for sequence, page_position in enumerate(
        ((100, 100), (105, 100), (130, 100)),
        start=1,
    ):
        observation = observer.observe(
            _frame(sequence, lobby, page_position=page_position)
        )
        snapshots.append(
            controller.step(observation, now_ns=observation.captured_at_ns)
        )

    assert all(snapshot.action is None for snapshot in snapshots)


def test_scene_template_requires_explicit_action_region() -> None:
    page = _pattern(901)
    action = _pattern(902)

    try:
        MenuSceneTemplate(
            "unsafe-action",
            MenuScene.LOBBY,
            _detector("page", page),
            _detector("action", action),
            None,
        )
    except ValueError as error:
        assert "动作区域" in str(error)
    else:
        raise AssertionError("动作模板缺少允许点击区域时必须被拒绝")


def test_scene_template_rejects_same_detector_for_page_and_action() -> None:
    pattern = _pattern(903)
    detector = _detector("shared", pattern)

    with pytest.raises(ValueError, match="页面锚点和动作锚点必须独立"):
        MenuSceneTemplate(
            "shared-anchor",
            MenuScene.LOBBY,
            detector,
            detector,
            ACTION_REGION,
        )


@pytest.mark.parametrize(
    ("action_accepted", "action_point", "page_point", "message"),
    [
        (False, (100.0, 100.0), (114.0, 110.0), "动作接受状态"),
        (True, None, (114.0, 110.0), "动作接受状态"),
        (True, (100.0, 100.0), (float("nan"), 110.0), "页面锚点"),
    ],
)
def test_scene_observation_rejects_inconsistent_or_non_finite_points(
    action_accepted: bool,
    action_point: tuple[float, float] | None,
    page_point: tuple[float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SceneObservation(
            frame_sequence=1,
            captured_at_ns=100_000_000,
            scene=MenuScene.LOBBY,
            candidate_scene=MenuScene.LOBBY,
            confidence=0.99,
            runner_up_confidence=0.1,
            accepted=True,
            reason=SceneDecisionReason.ACCEPTED,
            action_accepted=action_accepted,
            action_point=action_point,
            page_point=page_point,
            page_template_id="lobby",
        )
