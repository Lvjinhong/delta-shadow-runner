import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from delta_vision.external_loop import (
    CleanupSessionResult,
    ExternalLoopPhase,
    ExternalLoopSettings,
    ExternalLoopStatus,
    run_external_loop,
)
from delta_vision.game_session import WindowsMenuSettings
from delta_vision.menu_automation import (
    MenuActionKind,
    MenuControllerStatus,
    MenuScene,
    MenuTransition,
)
from delta_vision.menu_worker import MenuLoopResult
from delta_vision.navigation import NavigationStatus
from delta_vision.passive_scene import (
    PassiveReturnPolicy,
    PassiveReturnResult,
    PassiveReturnStatus,
)
from delta_vision.warehouse_cleanup import (
    CleanupActionIntent,
    CleanupIntentKind,
    SafeSlotState,
    WarehouseCleanupController,
    WarehouseCleanupPhase,
    WarehouseCleanupPolicy,
    WarehouseCleanupStatus,
    WarehouseObservation,
    WarehouseScene,
)
from delta_vision.worker import ControlLoopResult


class ObservationFactory:
    def __init__(self) -> None:
        self.sequence = 0
        self.captured_at_ns = 100_000_000

    def next(
        self,
        scene: WarehouseScene,
        *,
        safe_box_count: int | None = None,
        slots: tuple[SafeSlotState, SafeSlotState] = (
            SafeSlotState.UNKNOWN,
            SafeSlotState.UNKNOWN,
        ),
        open_point: tuple[float, float] | None = None,
        return_point: tuple[float, float] | None = None,
        transfer_points: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
        ] = (None, None),
        accepted: bool = True,
        frame_size: tuple[int, int] = (1920, 1080),
        advance_ms: int = 100,
    ) -> WarehouseObservation:
        observation = WarehouseObservation(
            frame_sequence=self.sequence,
            captured_at_ns=self.captured_at_ns,
            frame_size=frame_size,
            scene=scene,
            accepted=accepted,
            safe_box_count=safe_box_count,
            slots=slots,
            open_warehouse_point=open_point,
            return_base_point=return_point,
            transfer_points=transfer_points,
        )
        self.sequence += 1
        self.captured_at_ns += advance_ms * 1_000_000
        return observation


def _step(
    controller: WarehouseCleanupController,
    observation: WarehouseObservation,
    *,
    age_ms: int = 20,
):
    return controller.step(
        observation,
        now_ns=observation.captured_at_ns + age_ms * 1_000_000,
    )


def _confirm_base(
    controller: WarehouseCleanupController,
    factory: ObservationFactory,
    *,
    point: tuple[float, float] | None,
):
    return [
        _step(
            controller,
            factory.next(WarehouseScene.BASE, open_point=point),
        )
        for _ in range(3)
    ]


def _confirm_warehouse(
    controller: WarehouseCleanupController,
    factory: ObservationFactory,
    *,
    count: int,
    slots: tuple[SafeSlotState, SafeSlotState],
    return_point: tuple[float, float] | None = (190.0, 55.0),
    transfer_points: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
    ] = (None, None),
):
    return [
        _step(
            controller,
            factory.next(
                WarehouseScene.WAREHOUSE,
                safe_box_count=count,
                slots=slots,
                return_point=return_point,
                transfer_points=transfer_points,
            ),
        )
        for _ in range(3)
    ]


def test_empty_safe_box_opens_warehouse_skips_transfer_and_returns_base() -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy())
    frames = ObservationFactory()

    base = _confirm_base(controller, frames, point=(325.0, 55.0))
    warehouse = _confirm_warehouse(
        controller,
        frames,
        count=0,
        slots=(SafeSlotState.EMPTY, SafeSlotState.EMPTY),
    )
    final_base = _confirm_base(controller, frames, point=None)

    assert base[-1].action is not None
    assert base[-1].action.kind is CleanupIntentKind.OPEN_WAREHOUSE
    assert warehouse[-1].action is not None
    assert warehouse[-1].action.kind is CleanupIntentKind.RETURN_BASE
    assert all(
        snapshot.action is None or snapshot.action.kind is not CleanupIntentKind.TRANSFER_SLOT
        for snapshot in (*base, *warehouse, *final_base)
    )
    assert final_base[-1].status is WarehouseCleanupStatus.COMPLETED
    assert final_base[-1].phase is WarehouseCleanupPhase.COMPLETED


def test_action_expiry_is_bound_to_source_frame_capture_time() -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy())
    frames = ObservationFactory()
    result = None
    source = None
    for _ in range(3):
        source = frames.next(WarehouseScene.BASE, open_point=(325.0, 55.0))
        result = _step(controller, source, age_ms=249)

    assert source is not None
    assert result is not None and result.action is not None
    assert result.action.expires_at_ns == source.captured_at_ns + 250_000_000


def test_source_page_may_linger_after_open_and_return_actions() -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy())
    frames = ObservationFactory()
    _confirm_base(controller, frames, point=(325.0, 55.0))

    lingering_base = _step(
        controller,
        frames.next(WarehouseScene.BASE, open_point=(325.0, 55.0)),
    )
    warehouse = _confirm_warehouse(
        controller,
        frames,
        count=0,
        slots=(SafeSlotState.EMPTY, SafeSlotState.EMPTY),
    )
    lingering_warehouse = _step(
        controller,
        frames.next(
            WarehouseScene.WAREHOUSE,
            safe_box_count=0,
            slots=(SafeSlotState.EMPTY, SafeSlotState.EMPTY),
            return_point=(190.0, 55.0),
        ),
    )
    final_base = _confirm_base(controller, frames, point=None)

    assert lingering_base.status is WarehouseCleanupStatus.OBSERVING
    assert lingering_base.action is None
    assert warehouse[-1].action is not None
    assert lingering_warehouse.status is WarehouseCleanupStatus.OBSERVING
    assert lingering_warehouse.action is None
    assert final_base[-1].status is WarehouseCleanupStatus.COMPLETED


def test_confirmation_accepts_action_point_drift_within_policy() -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy())
    frames = ObservationFactory()

    results = [
        _step(
            controller,
            frames.next(WarehouseScene.BASE, open_point=point),
        )
        for point in ((325.0, 55.0), (328.0, 56.0), (330.0, 54.0))
    ]

    assert results[-1].action is not None
    assert results[-1].action.kind is CleanupIntentKind.OPEN_WAREHOUSE


def test_cleanup_action_intent_requires_real_position() -> None:
    with pytest.raises(ValueError, match="位置"):
        CleanupActionIntent(
            intent_id="invalid",
            kind=CleanupIntentKind.OPEN_WAREHOUSE,
            position=None,
            expires_at_ns=1,
        )


def test_nonempty_safe_box_without_verified_capability_stops() -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy())
    frames = ObservationFactory()
    _confirm_base(controller, frames, point=(325.0, 55.0))

    results = _confirm_warehouse(
        controller,
        frames,
        count=1,
        slots=(SafeSlotState.OCCUPIED, SafeSlotState.EMPTY),
        transfer_points=((850.0, 900.0), None),
    )

    assert results[-1].status is WarehouseCleanupStatus.STOPPED
    assert "未验证" in (results[-1].reason or "")
    assert results[-1].action is None


@pytest.mark.parametrize(
    "observation",
    [
        {"scene": WarehouseScene.UNKNOWN},
        {"scene": WarehouseScene.BASE, "frame_size": (2560, 1440)},
        {"scene": WarehouseScene.BASE, "accepted": False},
        {"scene": WarehouseScene.BASE, "open_point": None},
    ],
)
def test_unknown_ambiguous_size_or_missing_action_stops_without_intent(
    observation: dict[str, object],
) -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy())
    frames = ObservationFactory()
    values = {"scene": WarehouseScene.BASE, "open_point": (325.0, 55.0), **observation}

    result = _step(controller, frames.next(**values))

    assert result.status is WarehouseCleanupStatus.STOPPED
    assert result.action is None


def test_counter_and_slot_observations_must_agree() -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy())
    frames = ObservationFactory()
    _confirm_base(controller, frames, point=(325.0, 55.0))

    results = _confirm_warehouse(
        controller,
        frames,
        count=0,
        slots=(SafeSlotState.OCCUPIED, SafeSlotState.EMPTY),
    )

    assert results[-1].status is WarehouseCleanupStatus.STOPPED
    assert "不一致" in (results[-1].reason or "")


def test_two_items_transfer_one_at_a_time_after_strict_postcondition() -> None:
    policy = WarehouseCleanupPolicy(nonempty_transfer_verified=True)
    controller = WarehouseCleanupController(policy)
    frames = ObservationFactory()
    _confirm_base(controller, frames, point=(325.0, 55.0))

    occupied = _confirm_warehouse(
        controller,
        frames,
        count=2,
        slots=(SafeSlotState.OCCUPIED, SafeSlotState.OCCUPIED),
        transfer_points=((850.0, 900.0), (950.0, 900.0)),
    )
    unchanged = _step(
        controller,
        frames.next(
            WarehouseScene.WAREHOUSE,
            safe_box_count=2,
            slots=(SafeSlotState.OCCUPIED, SafeSlotState.OCCUPIED),
            return_point=(190.0, 55.0),
            transfer_points=((850.0, 900.0), (950.0, 900.0)),
        ),
    )
    one_left = _confirm_warehouse(
        controller,
        frames,
        count=1,
        slots=(SafeSlotState.EMPTY, SafeSlotState.OCCUPIED),
        transfer_points=(None, (950.0, 900.0)),
    )
    empty = _confirm_warehouse(
        controller,
        frames,
        count=0,
        slots=(SafeSlotState.EMPTY, SafeSlotState.EMPTY),
    )
    final_base = _confirm_base(controller, frames, point=None)

    assert occupied[-1].action is not None
    assert occupied[-1].action.intent_id == "transfer-safe-slot-0"
    assert unchanged.action is None
    assert one_left[-1].action is not None
    assert one_left[-1].action.intent_id == "transfer-safe-slot-1"
    assert empty[-1].action is not None
    assert empty[-1].action.kind is CleanupIntentKind.RETURN_BASE
    assert final_base[-1].status is WarehouseCleanupStatus.COMPLETED


def test_second_transfer_requires_fresh_stable_action_point() -> None:
    controller = WarehouseCleanupController(WarehouseCleanupPolicy(nonempty_transfer_verified=True))
    frames = ObservationFactory()
    _confirm_base(controller, frames, point=(325.0, 55.0))
    _confirm_warehouse(
        controller,
        frames,
        count=2,
        slots=(SafeSlotState.OCCUPIED, SafeSlotState.OCCUPIED),
        transfer_points=((850.0, 900.0), (950.0, 900.0)),
    )

    jittered = []
    for point in ((950.0, 900.0), (980.0, 900.0), (950.0, 900.0)):
        jittered.append(
            _step(
                controller,
                frames.next(
                    WarehouseScene.WAREHOUSE,
                    safe_box_count=1,
                    slots=(SafeSlotState.EMPTY, SafeSlotState.OCCUPIED),
                    return_point=(190.0, 55.0),
                    transfer_points=(None, point),
                ),
            )
        )

    stable = [
        _step(
            controller,
            frames.next(
                WarehouseScene.WAREHOUSE,
                safe_box_count=1,
                slots=(SafeSlotState.EMPTY, SafeSlotState.OCCUPIED),
                return_point=(190.0, 55.0),
                transfer_points=(None, (950.0, 900.0)),
            ),
        )
        for _ in range(2)
    ]

    assert all(snapshot.action is None for snapshot in jittered)
    assert stable[-1].action is not None
    assert stable[-1].action.intent_id == "transfer-safe-slot-1"


def test_transfer_without_counter_decrease_times_out_without_retry() -> None:
    policy = WarehouseCleanupPolicy(
        nonempty_transfer_verified=True,
        transition_timeout_ms=500,
    )
    controller = WarehouseCleanupController(policy)
    frames = ObservationFactory()
    _confirm_base(controller, frames, point=(325.0, 55.0))
    occupied = _confirm_warehouse(
        controller,
        frames,
        count=1,
        slots=(SafeSlotState.OCCUPIED, SafeSlotState.EMPTY),
        transfer_points=((850.0, 900.0), None),
    )
    emitted = [snapshot.action for snapshot in occupied if snapshot.action is not None]

    for _ in range(6):
        result = _step(
            controller,
            frames.next(
                WarehouseScene.WAREHOUSE,
                safe_box_count=1,
                slots=(SafeSlotState.OCCUPIED, SafeSlotState.EMPTY),
                transfer_points=((850.0, 900.0), None),
            ),
        )
        if result.status is WarehouseCleanupStatus.STOPPED:
            break
        if result.action is not None:
            emitted.append(result.action)

    assert result.status is WarehouseCleanupStatus.STOPPED
    assert "超时" in (result.reason or "")
    assert len(emitted) == 1


def _external_settings(*, cycle_limit: int) -> ExternalLoopSettings:
    profile = SimpleNamespace(
        transitions=(
            MenuTransition(
                source=MenuScene.BASE,
                target=MenuScene.IN_MATCH,
                action_kind=MenuActionKind.CLICK,
            ),
        ),
        frame_size=(1920, 1080),
    )
    entry = WindowsMenuSettings(
        target_window_title="三角洲行动  ",
        capture_backend="mss",
        emergency_virtual_key=123,
        max_key_hold_ms=250,
        menu_profile=profile,
        loop_interval_ms=20,
        max_duration_seconds=180,
    )
    match = SimpleNamespace(
        armed_ready=True,
        target_window_title="三角洲行动  ",
        emergency_virtual_key=123,
        perception=SimpleNamespace(frame_size=(1920, 1080)),
    )
    return ExternalLoopSettings(
        entry=entry,
        match=match,
        return_policy=PassiveReturnPolicy(max_duration_seconds=180),
        cycle_limit=cycle_limit,
    )


def _entry_result() -> MenuLoopResult:
    return MenuLoopResult(
        status=MenuControllerStatus.COMPLETED,
        frame_count=3,
        action_count=1,
        duration_ns=1,
        reason=None,
        terminal_scene=MenuScene.IN_MATCH,
    )


def _match_result() -> ControlLoopResult:
    return ControlLoopResult(
        status=NavigationStatus.STOPPED,
        frame_count=3,
        duration_ns=1,
        reason="returned",
    )


def _return_result() -> PassiveReturnResult:
    return PassiveReturnResult(
        status=PassiveReturnStatus.BASE_CONFIRMED,
        terminal_scene=MenuScene.BASE,
        seen_scenes=(MenuScene.BASE,),
        frame_count=3,
        duration_ns=1,
        reason=None,
    )


class FakeCleanupSession:
    def __init__(self, results: list[CleanupSessionResult]) -> None:
        self.results = results
        self.calls: list[tuple[Path, bool, str]] = []

    def run(self, *, artifacts: Path, armed: bool, run_id: str) -> CleanupSessionResult:
        self.calls.append((artifacts, armed, run_id))
        return self.results.pop(0)


def test_cleanup_stopped_prevents_next_external_loop_cycle(tmp_path: Path) -> None:
    session = FakeCleanupSession(
        [CleanupSessionResult(completed=False, reason="保险箱状态未知", summary={"count": None})]
    )
    entry_count = 0

    def entry_runner(**_kwargs: object) -> MenuLoopResult:
        nonlocal entry_count
        entry_count += 1
        return _entry_result()

    result = run_external_loop(
        _external_settings(cycle_limit=2),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="cleanup-stopped",
        entry_runner=entry_runner,
        match_runner=lambda *_args, **_kwargs: _match_result(),
        return_observer=lambda **_kwargs: _return_result(),
        cleanup_session=session,
    )

    assert result.status is ExternalLoopStatus.STOPPED
    assert result.stopped_phase is ExternalLoopPhase.CLEANUP
    assert result.completed_cycles == 0
    assert entry_count == 1
    assert result.cycles[0].cleanup_started is True
    assert result.cycles[0].cleanup_completed is False
    summary = json.loads((tmp_path / "loop/external-loop-summary.json").read_text())
    assert summary["cycles"][0]["cleanup_summary"] == {"count": None}


def test_cleanup_completed_allows_next_external_loop_cycle(tmp_path: Path) -> None:
    session = FakeCleanupSession(
        [
            CleanupSessionResult(completed=True, reason=None, summary={"count": 0}),
            CleanupSessionResult(completed=True, reason=None, summary={"count": 0}),
        ]
    )

    result = run_external_loop(
        _external_settings(cycle_limit=2),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="cleanup-complete",
        entry_runner=lambda **_kwargs: _entry_result(),
        match_runner=lambda *_args, **_kwargs: _match_result(),
        return_observer=lambda **_kwargs: _return_result(),
        cleanup_session=session,
    )

    assert result.status is ExternalLoopStatus.COMPLETED
    assert result.completed_cycles == 2
    assert all(cycle.cleanup_completed for cycle in result.cycles)
    assert len(session.calls) == 2
