import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from delta_vision.external_loop import (
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
from delta_vision.worker import ControlLoopResult


def _settings(*, cycle_limit: int = 1) -> ExternalLoopSettings:
    menu_profile = SimpleNamespace(
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
        menu_profile=menu_profile,
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


def _entry(
    *,
    status: MenuControllerStatus = MenuControllerStatus.COMPLETED,
    scene: MenuScene = MenuScene.IN_MATCH,
) -> MenuLoopResult:
    return MenuLoopResult(
        status=status,
        frame_count=10,
        action_count=1,
        duration_ns=1_000_000,
        reason=None,
        terminal_scene=scene,
    )


def _match() -> ControlLoopResult:
    return ControlLoopResult(
        status=NavigationStatus.STOPPED,
        frame_count=20,
        duration_ns=2_000_000,
        reason="等待被动返回确认",
    )


def _returned(
    status: PassiveReturnStatus = PassiveReturnStatus.BASE_CONFIRMED,
) -> PassiveReturnResult:
    return PassiveReturnResult(
        status=status,
        terminal_scene=(
            MenuScene.BASE if status is PassiveReturnStatus.BASE_CONFIRMED else MenuScene.UNKNOWN
        ),
        seen_scenes=(MenuScene.POST_MATCH, MenuScene.BASE),
        frame_count=12,
        duration_ns=3_000_000,
        reason=None if status is PassiveReturnStatus.BASE_CONFIRMED else "超时",
    )


def test_external_loop_completes_one_full_cycle(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def entry_runner(**kwargs: object) -> MenuLoopResult:
        calls.append(("entry", str(kwargs["run_id"])))
        return _entry()

    def match_runner(_settings: object, **kwargs: object) -> ControlLoopResult:
        calls.append(("match", str(kwargs["run_id"])))
        return _match()

    def return_observer(**kwargs: object) -> PassiveReturnResult:
        calls.append(("return", str(kwargs["run_id"])))
        return _returned()

    def cleanup_hook(cycle: object) -> None:
        calls.append(("cleanup", cycle.run_id))

    result = run_external_loop(
        _settings(),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="loop-01",
        entry_runner=entry_runner,
        match_runner=match_runner,
        return_observer=return_observer,
        cleanup_hook=cleanup_hook,
    )

    assert result.status is ExternalLoopStatus.COMPLETED
    assert result.completed_cycles == 1
    assert result.cycles[0].cleanup_ran is True
    assert calls == [
        ("entry", "loop-01-c0001"),
        ("match", "loop-01-c0001"),
        ("return", "loop-01-c0001"),
        ("cleanup", "loop-01-c0001"),
    ]
    summary = json.loads((tmp_path / "loop" / "external-loop-summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["completed_cycles"] == 1


def test_second_entry_waits_for_first_base_and_cleanup(tmp_path: Path) -> None:
    order: list[str] = []

    result = run_external_loop(
        _settings(cycle_limit=2),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="loop-02",
        entry_runner=lambda **_kwargs: order.append("entry") or _entry(),
        match_runner=lambda _settings, **_kwargs: order.append("match") or _match(),
        return_observer=lambda **_kwargs: order.append("return") or _returned(),
        cleanup_hook=lambda _cycle: order.append("cleanup"),
    )

    assert result.status is ExternalLoopStatus.COMPLETED
    assert order == [
        "entry",
        "match",
        "return",
        "cleanup",
        "entry",
        "match",
        "return",
        "cleanup",
    ]


def test_entry_stopped_never_starts_other_phases(tmp_path: Path) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("不应进入后续阶段")

    result = run_external_loop(
        _settings(),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="loop-03",
        entry_runner=lambda **_kwargs: _entry(status=MenuControllerStatus.STOPPED),
        match_runner=forbidden,
        return_observer=forbidden,
        cleanup_hook=forbidden,
    )

    assert result.status is ExternalLoopStatus.STOPPED
    assert result.completed_cycles == 0
    assert result.stopped_phase is ExternalLoopPhase.ENTERING


def test_completed_entry_without_in_match_stops_closed(tmp_path: Path) -> None:
    result = run_external_loop(
        _settings(),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="loop-04",
        entry_runner=lambda **_kwargs: _entry(scene=MenuScene.BASE),
        match_runner=lambda *_args, **_kwargs: pytest.fail("不应启动局内 runner"),
        return_observer=lambda **_kwargs: pytest.fail("不应观察返回"),
    )

    assert result.status is ExternalLoopStatus.STOPPED
    assert result.stopped_phase is ExternalLoopPhase.ENTERING
    assert "IN_MATCH" in (result.reason or "")


def test_match_stopped_still_enters_passive_return(tmp_path: Path) -> None:
    return_calls = 0

    def return_observer(**_kwargs: object) -> PassiveReturnResult:
        nonlocal return_calls
        return_calls += 1
        return _returned()

    result = run_external_loop(
        _settings(),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="loop-05",
        entry_runner=lambda **_kwargs: _entry(),
        match_runner=lambda _settings, **_kwargs: _match(),
        return_observer=return_observer,
    )

    assert result.status is ExternalLoopStatus.COMPLETED
    assert return_calls == 1


def test_return_timeout_never_runs_cleanup_or_next_cycle(tmp_path: Path) -> None:
    cleanup_calls = 0

    def cleanup_hook(_cycle: object) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    result = run_external_loop(
        _settings(cycle_limit=2),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="loop-06",
        entry_runner=lambda **_kwargs: _entry(),
        match_runner=lambda _settings, **_kwargs: _match(),
        return_observer=lambda **_kwargs: _returned(PassiveReturnStatus.STOPPED),
        cleanup_hook=cleanup_hook,
    )

    assert result.status is ExternalLoopStatus.STOPPED
    assert result.stopped_phase is ExternalLoopPhase.RETURN_OBSERVING
    assert cleanup_calls == 0


def test_base_status_with_wrong_terminal_never_runs_cleanup(tmp_path: Path) -> None:
    cleanup_calls = 0

    def cleanup_hook(_cycle: object) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    inconsistent_return = SimpleNamespace(
        status=PassiveReturnStatus.BASE_CONFIRMED,
        terminal_scene=MenuScene.UNKNOWN,
        seen_scenes=(MenuScene.BASE,),
        frame_count=3,
        duration_ns=3_000_000,
        reason=None,
    )
    result = run_external_loop(
        _settings(),
        artifacts=tmp_path / "loop",
        armed=True,
        run_id="loop-inconsistent-return",
        entry_runner=lambda **_kwargs: _entry(),
        match_runner=lambda _settings, **_kwargs: _match(),
        return_observer=lambda **_kwargs: inconsistent_return,
        cleanup_hook=cleanup_hook,
    )

    assert result.status is ExternalLoopStatus.STOPPED
    assert result.stopped_phase is ExternalLoopPhase.RETURN_OBSERVING
    assert "BASE" in (result.reason or "")
    assert cleanup_calls == 0


def test_match_exception_writes_failure_summary(tmp_path: Path) -> None:
    root = tmp_path / "loop"

    with pytest.raises(RuntimeError, match="match boom"):
        run_external_loop(
            _settings(),
            artifacts=root,
            armed=True,
            run_id="loop-07",
            entry_runner=lambda **_kwargs: _entry(),
            match_runner=lambda _settings, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("match boom")
            ),
            return_observer=lambda **_kwargs: pytest.fail("不应观察返回"),
        )

    summary = json.loads((root / "external-loop-summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed_phase"] == "match_running"
    assert summary["cycles"][0]["entry"]["terminal_scene"] == "in_match"


def test_cleanup_exception_records_started_but_not_completed(tmp_path: Path) -> None:
    root = tmp_path / "loop"

    def partial_cleanup(_cycle: object) -> None:
        raise RuntimeError("cleanup boom")

    with pytest.raises(RuntimeError, match="cleanup boom"):
        run_external_loop(
            _settings(),
            artifacts=root,
            armed=True,
            run_id="loop-cleanup-failure",
            entry_runner=lambda **_kwargs: _entry(),
            match_runner=lambda _settings, **_kwargs: _match(),
            return_observer=lambda **_kwargs: _returned(),
            cleanup_hook=partial_cleanup,
        )

    summary = json.loads((root / "external-loop-summary.json").read_text())
    assert summary["failed_phase"] == "cleanup"
    assert summary["cycles"][0]["cleanup_started"] is True
    assert summary["cycles"][0]["cleanup_completed"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_window_title", "其他窗口", "窗口标题"),
        ("emergency_virtual_key", 122, "急停键"),
    ],
)
def test_entry_and_match_must_share_window_safety_binding(
    field: str,
    value: object,
    message: str,
) -> None:
    settings = _settings()
    match_values = vars(settings.match).copy()
    match_values[field] = value

    with pytest.raises(ValueError, match=message):
        ExternalLoopSettings(
            entry=settings.entry,
            match=SimpleNamespace(**match_values),
            return_policy=settings.return_policy,
            cycle_limit=1,
        )


def test_entry_contract_requires_one_base_click_to_in_match() -> None:
    settings = _settings()
    bad_profile = SimpleNamespace(
        transitions=(
            MenuTransition(
                source=MenuScene.BASE,
                target=MenuScene.IN_MATCH,
                action_kind=MenuActionKind.KEY,
                key="space",
            ),
        ),
        frame_size=(1920, 1080),
    )
    bad_entry = WindowsMenuSettings(
        target_window_title=settings.entry.target_window_title,
        capture_backend="mss",
        emergency_virtual_key=123,
        max_key_hold_ms=250,
        menu_profile=bad_profile,
        loop_interval_ms=20,
        max_duration_seconds=180,
    )

    with pytest.raises(ValueError, match=r"BASE.*CLICK.*IN_MATCH"):
        ExternalLoopSettings(
            entry=bad_entry,
            match=settings.match,
            return_policy=settings.return_policy,
            cycle_limit=1,
        )
