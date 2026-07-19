import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from delta_vision.config import CaptureRegion
from delta_vision.events import JsonlEventWriter
from delta_vision.external_loop import CleanupSessionResult
from delta_vision.frames import CapturedFrame, FrameRecorder
from delta_vision.menu_automation import (
    MenuActionKind,
    MenuScene,
    MenuTransition,
    SceneDecisionReason,
    SceneObservation,
)
from delta_vision.safe_input import Win32InputActuator
from delta_vision.warehouse_cleanup import (
    WAREHOUSE_BASE_TEMPLATE_ID,
    WAREHOUSE_EMPTY_EVIDENCE_IDS,
    WAREHOUSE_EMPTY_TEMPLATE_ID,
    CleanupActionIntent,
    CleanupIntentKind,
    MenuProfileWarehouseObserver,
    SafeSlotState,
    WarehouseCleanupController,
    WarehouseCleanupPolicy,
    WarehouseCleanupStatus,
    WarehouseObservation,
    WarehouseScene,
)
from delta_vision.warehouse_worker import (
    ArmedCleanupExecutor,
    DryRunCleanupExecutor,
    DuplicateCleanupActionError,
    ExpiredCleanupActionError,
    WarehouseCleanupLoopResult,
    WindowsWarehouseCleanupSession,
    WindowsWarehouseCleanupSettings,
    build_windows_warehouse_cleanup_runtime,
    run_warehouse_cleanup_loop,
    run_windows_warehouse_cleanup,
)


class MutableClock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns


class SequenceSource:
    def __init__(self, scenes: list[WarehouseScene], clock: MutableClock) -> None:
        self.frames = [
            CapturedFrame(
                sequence=index,
                captured_at_ns=100_000_000 + index * 100_000_000,
                image=np.full((1080, 1920, 3), index, dtype=np.uint8),
                source="warehouse-replay",
                metadata={"scene": str(scene)},
            )
            for index, scene in enumerate(scenes)
        ]
        self.clock = clock
        self.closed = False

    def grab(self) -> CapturedFrame | None:
        if not self.frames:
            return None
        frame = self.frames.pop(0)
        self.clock.now_ns = frame.captured_at_ns + 20_000_000
        return frame

    def close(self) -> None:
        self.closed = True


class MetadataObserver:
    def observe(self, frame: CapturedFrame) -> WarehouseObservation:
        scene = WarehouseScene(frame.metadata["scene"])
        if scene is WarehouseScene.BASE:
            return WarehouseObservation(
                frame_sequence=frame.sequence,
                captured_at_ns=frame.captured_at_ns,
                frame_size=(1920, 1080),
                scene=scene,
                accepted=True,
                open_warehouse_point=(322.5, 55.0),
            )
        if scene is WarehouseScene.WAREHOUSE:
            return WarehouseObservation(
                frame_sequence=frame.sequence,
                captured_at_ns=frame.captured_at_ns,
                frame_size=(1920, 1080),
                scene=scene,
                accepted=True,
                safe_box_count=0,
                slots=(SafeSlotState.EMPTY, SafeSlotState.EMPTY),
                return_base_point=(192.5, 55.0),
            )
        return WarehouseObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            frame_size=(1920, 1080),
            scene=WarehouseScene.UNKNOWN,
            accepted=False,
        )


def _run(tmp_path: Path, scenes: list[WarehouseScene]):
    clock = MutableClock()
    source = SequenceSource(scenes, clock)
    executor = DryRunCleanupExecutor(
        capture_region=CaptureRegion(left=320, top=156, width=1920, height=1080),
        clock_ns=clock,
    )
    result = run_warehouse_cleanup_loop(
        source=source,
        observer=MetadataObserver(),
        controller=WarehouseCleanupController(WarehouseCleanupPolicy()),
        executor=executor,
        recorder=FrameRecorder(tmp_path / "replay"),
        event_writer=JsonlEventWriter(
            tmp_path / "events.jsonl",
            run_id="warehouse-dry-run",
            truncate=True,
        ),
        loop_interval_ms=20,
        max_duration_seconds=10,
        clock_ns=clock,
        sleep_fn=lambda _seconds: None,
    )
    return result, source, executor


def test_realistic_empty_cleanup_replay_completes_without_os_input(tmp_path: Path) -> None:
    result, source, executor = _run(
        tmp_path,
        [WarehouseScene.BASE] * 3
        + [WarehouseScene.WAREHOUSE] * 3
        + [WarehouseScene.BASE] * 3,
    )

    assert result.status is WarehouseCleanupStatus.COMPLETED
    assert result.frame_count == 9
    assert result.action_count == 2
    assert result.safe_box_count == 0
    assert source.closed is True
    assert [record.intent_id for record in executor.records] == [
        "open-warehouse",
        "return-base",
    ]
    assert [record.local_position for record in executor.records] == [
        (322, 55),
        (192, 55),
    ]
    assert [record.screen_position for record in executor.records] == [
        (642, 211),
        (512, 211),
    ]
    assert not (tmp_path / "replay/input-events.jsonl").exists()

    frame_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "warehouse_frame"
    ]
    assert len(frame_events) == 9
    assert frame_events[-1]["payload"]["status"] == "completed"


def test_unknown_frame_stops_without_dry_run_action(tmp_path: Path) -> None:
    result, source, executor = _run(tmp_path, [WarehouseScene.UNKNOWN])

    assert result.status is WarehouseCleanupStatus.STOPPED
    assert result.action_count == 0
    assert "未知" in (result.reason or "")
    assert executor.records == ()
    assert source.closed is True


def test_dry_run_executor_rejects_expired_and_duplicate_intents() -> None:
    clock = MutableClock()
    executor = DryRunCleanupExecutor(
        capture_region=CaptureRegion(0, 0, 1920, 1080),
        clock_ns=clock,
    )
    controller = WarehouseCleanupController(
        WarehouseCleanupPolicy(confirmation_frames=1)
    )
    observation = WarehouseObservation(
        frame_sequence=0,
        captured_at_ns=100,
        frame_size=(1920, 1080),
        scene=WarehouseScene.BASE,
        accepted=True,
        open_warehouse_point=(322.5, 55.0),
    )
    snapshot = controller.step(observation, now_ns=100)
    assert snapshot.action is not None

    with pytest.raises(ExpiredCleanupActionError):
        executor.execute(snapshot.action, now_ns=snapshot.action.expires_at_ns)

    executor.execute(snapshot.action, now_ns=101)
    with pytest.raises(DuplicateCleanupActionError):
        executor.execute(snapshot.action, now_ns=102)


def test_source_close_failure_is_not_retried(tmp_path: Path) -> None:
    clock = MutableClock()

    class FailingCloseSource(SequenceSource):
        def __init__(self) -> None:
            super().__init__([WarehouseScene.UNKNOWN], clock)
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("close-failure")

    source = FailingCloseSource()

    with pytest.raises(RuntimeError, match="close-failure"):
        run_warehouse_cleanup_loop(
            source=source,
            observer=MetadataObserver(),
            controller=WarehouseCleanupController(WarehouseCleanupPolicy()),
            executor=DryRunCleanupExecutor(
                capture_region=CaptureRegion(0, 0, 1920, 1080),
                clock_ns=clock,
            ),
            recorder=FrameRecorder(tmp_path / "replay"),
            event_writer=JsonlEventWriter(
                tmp_path / "events.jsonl",
                run_id="close-failure",
                truncate=True,
            ),
            loop_interval_ms=20,
            max_duration_seconds=10,
            clock_ns=clock,
            sleep_fn=lambda _seconds: None,
        )

    assert source.close_calls == 1


class WarehouseMenuSceneObserver:
    def observe(self, frame: CapturedFrame) -> SceneObservation:
        scene = MenuScene(frame.metadata["scene"])
        accepted = scene in {MenuScene.BASE, MenuScene.WAREHOUSE}
        action_point = (
            (322.5, 55.0) if scene is MenuScene.BASE else (192.5, 55.0)
        )
        template_id = (
            WAREHOUSE_BASE_TEMPLATE_ID
            if scene is MenuScene.BASE
            else WAREHOUSE_EMPTY_TEMPLATE_ID
        )
        return SceneObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            scene=scene if accepted else MenuScene.UNKNOWN,
            candidate_scene=scene if accepted else None,
            confidence=0.99 if accepted else 0.0,
            runner_up_confidence=0.1 if accepted else 0.0,
            accepted=accepted,
            reason=(
                SceneDecisionReason.ACCEPTED
                if accepted
                else SceneDecisionReason.BELOW_THRESHOLD
            ),
            action_accepted=accepted,
            action_point=action_point if accepted else None,
            page_point=(100.0, 100.0) if accepted else None,
            page_template_id=template_id if accepted else None,
        )


def _warehouse_profile(*, observer=None):
    evidence_ids = tuple(sorted(WAREHOUSE_EMPTY_EVIDENCE_IDS))
    provenance = (
        SimpleNamespace(
            template_id=WAREHOUSE_BASE_TEMPLATE_ID,
            scene=MenuScene.BASE,
            page_content_sha256="page-base",
            action_content_sha256="action-base",
            action_sha256="sha-action-base",
            evidence_ids=(),
            evidence_sha256=(),
            evidence_content_sha256=(),
        ),
        SimpleNamespace(
            template_id=WAREHOUSE_EMPTY_TEMPLATE_ID,
            scene=MenuScene.WAREHOUSE,
            page_content_sha256="page-empty",
            action_content_sha256="action-empty",
            action_sha256="sha-action-empty",
            evidence_ids=evidence_ids,
            evidence_sha256=tuple(f"sha-{item}" for item in evidence_ids),
            evidence_content_sha256=tuple(
                f"content-{item}" for item in evidence_ids
            ),
        ),
    )
    return SimpleNamespace(
        profile_id="warehouse-empty-test-v2",
        frame_size=(1920, 1080),
        observer=(
            SimpleNamespace(observe=lambda _frame: None)
            if observer is None
            else observer
        ),
        template_provenance=provenance,
        transitions=(
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
        ),
        stop_scenes=frozenset(),
    )


def _windows_cleanup_settings(*, armed_ready: bool = False, profile=None):
    return WindowsWarehouseCleanupSettings(
        target_window_title="三角洲行动  ",
        capture_backend="mss",
        emergency_virtual_key=123,
        max_key_hold_ms=250,
        menu_profile=_warehouse_profile() if profile is None else profile,
        loop_interval_ms=20,
        max_duration_seconds=10,
        armed_ready=armed_ready,
    )


class EmptyFrameSource:
    def __init__(self) -> None:
        self.closed = False

    def grab(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class MouseGateway:
    def __init__(self) -> None:
        self.title = "三角洲行动  "
        self.window_handle = 123
        self.emergency_pressed = False
        self.sent: list[tuple[object, ...]] = []

    def foreground_title(self) -> str:
        return self.title

    def foreground_window_handle(self) -> int:
        return self.window_handle

    def is_key_pressed(self, virtual_key: int) -> bool:
        assert virtual_key == 123
        return self.emergency_pressed

    def send_key(self, scan_code: int, *, key_up: bool) -> int:
        self.sent.append(("key", scan_code, key_up))
        return 1

    def send_mouse_relative(self, dx: int, dy: int) -> int:
        self.sent.append(("mouse", dx, dy))
        return 1

    def send_mouse_absolute(self, screen_x: int, screen_y: int) -> int:
        self.sent.append(("mouse_absolute", screen_x, screen_y))
        return 1

    def send_mouse_left(self, *, key_up: bool) -> int:
        self.sent.append(("mouse_left", key_up))
        return 1


@pytest.mark.parametrize("virtual_key", [122, 124, True])
def test_windows_warehouse_settings_requires_f12(virtual_key: object) -> None:
    with pytest.raises(ValueError, match="F12"):
        WindowsWarehouseCleanupSettings(
            target_window_title="三角洲行动  ",
            capture_backend="mss",
            emergency_virtual_key=virtual_key,  # type: ignore[arg-type]
            max_key_hold_ms=250,
            menu_profile=_warehouse_profile(),
            loop_interval_ms=20,
            max_duration_seconds=10,
        )


def test_windows_warehouse_builder_defaults_to_dry_run_without_gateway(
    tmp_path: Path,
) -> None:
    source = EmptyFrameSource()
    calls: list[str] = []

    runtime = build_windows_warehouse_cleanup_runtime(
        _windows_cleanup_settings(),
        artifacts=tmp_path,
        armed=False,
        run_id="warehouse-builder-dry-run",
        region_resolver=lambda _title: (
            calls.append("region") or CaptureRegion(320, 156, 1920, 1080)
        ),
        window_handle_resolver=lambda _title: calls.append("handle") or 123,
        mss_factory=lambda _region: calls.append("source") or source,
        gateway_factory=lambda: (_ for _ in ()).throw(
            AssertionError("DryRun 不应构造输入 gateway")
        ),
    )

    assert isinstance(runtime.observer, MenuProfileWarehouseObserver)
    assert isinstance(runtime.executor, DryRunCleanupExecutor)
    assert runtime.actuator is None
    assert runtime.source is source
    assert calls == ["handle", "region", "source"]
    source.close()


def test_windows_warehouse_builder_rejects_bad_resolution_before_side_effects(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="1920x1080"):
        build_windows_warehouse_cleanup_runtime(
            _windows_cleanup_settings(),
            artifacts=tmp_path,
            armed=False,
            region_resolver=lambda _title: CaptureRegion(0, 0, 2560, 1440),
            window_handle_resolver=lambda _title: 123,
            mss_factory=lambda _region: (_ for _ in ()).throw(
                AssertionError("分辨率不匹配时不应创建截图源")
            ),
        )


def test_armed_cleanup_requires_separate_ready_before_window_resolution(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="armed_ready=false"):
        build_windows_warehouse_cleanup_runtime(
            _windows_cleanup_settings(armed_ready=False),
            artifacts=tmp_path,
            armed=True,
            region_resolver=lambda _title: (_ for _ in ()).throw(
                AssertionError("未授权时不应解析窗口")
            ),
        )


def test_armed_builder_binds_mouse_only_actuator_and_audits_real_mode(
    tmp_path: Path,
) -> None:
    source = EmptyFrameSource()
    gateway = MouseGateway()
    clock = MutableClock()
    clock.now_ns = 100
    runtime = build_windows_warehouse_cleanup_runtime(
        _windows_cleanup_settings(armed_ready=True),
        artifacts=tmp_path,
        armed=True,
        run_id="warehouse-builder-armed",
        region_resolver=lambda _title: CaptureRegion(320, 156, 1920, 1080),
        window_handle_resolver=lambda _title: 123,
        mss_factory=lambda _region: source,
        gateway_factory=lambda: gateway,
        clock_ns=clock,
    )
    action = CleanupActionIntent(
        intent_id="open-warehouse",
        kind=CleanupIntentKind.OPEN_WAREHOUSE,
        position=(10.4, 20.6),
        expires_at_ns=200,
    )

    record = runtime.executor.execute(action, now_ns=100)

    assert isinstance(runtime.executor, ArmedCleanupExecutor)
    assert isinstance(runtime.actuator, Win32InputActuator)
    assert record.dry_run is False
    assert record.screen_position == (330, 177)
    assert gateway.sent == [
        ("mouse_absolute", 330, 177),
        ("mouse_left", False),
        ("mouse_left", True),
    ]
    with pytest.raises(DuplicateCleanupActionError):
        runtime.executor.execute(action, now_ns=101)
    with pytest.raises(ValueError, match="非空转移"):
        runtime.executor.execute(
            CleanupActionIntent(
                intent_id="transfer-safe-slot-0",
                kind=CleanupIntentKind.TRANSFER_SLOT,
                position=(100.0, 100.0),
                expires_at_ns=200,
                slot_index=0,
            ),
            now_ns=101,
        )
    assert len(gateway.sent) == 3
    source.close()


def test_armed_builder_derives_region_from_exact_safety_gate_handle(
    tmp_path: Path,
) -> None:
    source = EmptyFrameSource()
    resolved_handles: list[int] = []

    runtime = build_windows_warehouse_cleanup_runtime(
        _windows_cleanup_settings(armed_ready=True),
        artifacts=tmp_path,
        armed=True,
        window_handle_resolver=lambda _title: 222,
        region_resolver=lambda handle: (
            resolved_handles.append(handle)
            or CaptureRegion(100, 200, 1920, 1080)
        ),
        mss_factory=lambda _region: source,
        gateway_factory=MouseGateway,
        clock_ns=lambda: 100,
    )

    assert runtime.target_window_handle == 222
    assert resolved_handles == [222]
    source.close()


def test_armed_executor_consumes_failed_intent_and_releases_once() -> None:
    class FailingActuator:
        events: tuple[object, ...] = ()

        def __init__(self) -> None:
            self.click_calls = 0
            self.release_calls = 0

        def click_left_at(self, *_args, **_kwargs) -> None:
            self.click_calls += 1
            raise RuntimeError("partial-input-failure")

        def release_all(self, **_kwargs) -> None:
            self.release_calls += 1

    actuator = FailingActuator()
    executor = ArmedCleanupExecutor(
        actuator=actuator,
        capture_region=CaptureRegion(0, 0, 1920, 1080),
        clock_ns=lambda: 100,
    )
    action = CleanupActionIntent(
        intent_id="open-warehouse",
        kind=CleanupIntentKind.OPEN_WAREHOUSE,
        position=(10.0, 20.0),
        expires_at_ns=200,
    )

    with pytest.raises(RuntimeError, match="partial-input-failure"):
        executor.execute(action, now_ns=100)
    with pytest.raises(DuplicateCleanupActionError):
        executor.execute(action, now_ns=101)

    assert actuator.click_calls == 1
    assert actuator.release_calls == 1
    assert executor.records == ()


def test_armed_executor_rejects_expired_intent_without_actuator_call() -> None:
    class UntouchedActuator:
        events: tuple[object, ...] = ()

        def click_left_at(self, *_args, **_kwargs) -> None:
            raise AssertionError("过期动作不能触碰 actuator")

        def release_all(self, **_kwargs) -> None:
            raise AssertionError("过期动作不需要释放输入")

    executor = ArmedCleanupExecutor(
        actuator=UntouchedActuator(),
        capture_region=CaptureRegion(0, 0, 1920, 1080),
        clock_ns=lambda: 200,
    )

    with pytest.raises(ExpiredCleanupActionError):
        executor.execute(
            CleanupActionIntent(
                intent_id="open-warehouse",
                kind=CleanupIntentKind.OPEN_WAREHOUSE,
                position=(10.0, 20.0),
                expires_at_ns=200,
            ),
            now_ns=100,
        )


@pytest.mark.parametrize(
    ("status", "safe_box_count", "expected_completed"),
    [
        (WarehouseCleanupStatus.COMPLETED, 0, True),
        (WarehouseCleanupStatus.COMPLETED, None, False),
        (WarehouseCleanupStatus.STOPPED, 0, False),
        (WarehouseCleanupStatus.STOPPED, None, False),
    ],
)
def test_concrete_cleanup_session_requires_completed_zero_count(
    tmp_path: Path,
    status: WarehouseCleanupStatus,
    safe_box_count: int | None,
    expected_completed: bool,
) -> None:
    calls: list[dict[str, object]] = []

    def runner(settings, **kwargs) -> WarehouseCleanupLoopResult:
        calls.append({"settings": settings, **kwargs})
        return WarehouseCleanupLoopResult(
            status=status,
            frame_count=9,
            action_count=2,
            duration_ns=900,
            reason=None if status is WarehouseCleanupStatus.COMPLETED else "未知页面",
            safe_box_count=safe_box_count,
        )

    session = WindowsWarehouseCleanupSession(
        settings=_windows_cleanup_settings(armed_ready=True),
        runner=runner,
    )
    result = session.run(
        artifacts=tmp_path / "cleanup",
        armed=True,
        run_id="warehouse-cleanup-session",
    )

    assert isinstance(result, CleanupSessionResult)
    assert result.completed is expected_completed
    assert result.summary["armed"] is True
    assert result.summary["safe_box_count"] == safe_box_count
    assert calls[0]["artifacts"] == tmp_path / "cleanup/runtime"
    assert calls[0]["run_id"] == "warehouse-cleanup-session"
    summary = json.loads(
        (tmp_path / "cleanup/cleanup-summary.json").read_text(encoding="utf-8")
    )
    assert summary["completed"] is expected_completed


def test_empty_warehouse_armed_replay_emits_two_guarded_clicks_and_audit(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    source = SequenceSource(
        [WarehouseScene.BASE] * 3
        + [WarehouseScene.WAREHOUSE] * 3
        + [WarehouseScene.BASE] * 3,
        clock,
    )
    gateway = MouseGateway()
    settings = _windows_cleanup_settings(
        armed_ready=True,
        profile=_warehouse_profile(observer=WarehouseMenuSceneObserver()),
    )

    result = run_windows_warehouse_cleanup(
        settings,
        artifacts=tmp_path / "armed-runtime",
        armed=True,
        run_id="warehouse-empty-armed-replay",
        region_resolver=lambda _title: CaptureRegion(320, 156, 1920, 1080),
        window_handle_resolver=lambda _title: 123,
        mss_factory=lambda _region: source,
        gateway_factory=lambda: gateway,
        clock_ns=clock,
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is WarehouseCleanupStatus.COMPLETED
    assert result.safe_box_count == 0
    assert result.action_count == 2
    assert source.closed is True
    assert gateway.sent == [
        ("mouse_absolute", 642, 211),
        ("mouse_left", False),
        ("mouse_left", True),
        ("mouse_absolute", 512, 211),
        ("mouse_left", False),
        ("mouse_left", True),
    ]
    input_events = [
        json.loads(line)
        for line in (
            tmp_path / "armed-runtime/replay/input-events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(input_events) == 6
    warehouse_events = [
        json.loads(line)
        for line in (
            tmp_path / "armed-runtime/events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    executions = [
        event["payload"]["execution"]
        for event in warehouse_events
        if event["event_type"] == "warehouse_frame"
        and event["payload"]["execution"] is not None
    ]
    assert [execution["dry_run"] for execution in executions] == [False, False]


def test_unknown_warehouse_frame_stops_armed_runtime_without_input(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    source = SequenceSource([WarehouseScene.UNKNOWN], clock)
    gateway = MouseGateway()

    result = run_windows_warehouse_cleanup(
        _windows_cleanup_settings(
            armed_ready=True,
            profile=_warehouse_profile(observer=WarehouseMenuSceneObserver()),
        ),
        artifacts=tmp_path / "unknown-runtime",
        armed=True,
        run_id="warehouse-unknown-armed-replay",
        region_resolver=lambda _title: CaptureRegion(320, 156, 1920, 1080),
        window_handle_resolver=lambda _title: 123,
        mss_factory=lambda _region: source,
        gateway_factory=lambda: gateway,
        clock_ns=clock,
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is WarehouseCleanupStatus.STOPPED
    assert result.action_count == 0
    assert gateway.sent == []
    assert not (
        tmp_path / "unknown-runtime/replay/input-events.jsonl"
    ).exists()
