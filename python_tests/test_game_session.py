import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from delta_vision.config import CaptureRegion
from delta_vision.game_session import (
    GameSessionSettings,
    GameSessionStatus,
    build_windows_menu_runtime,
    load_game_session_settings,
    run_game_session,
)
from delta_vision.menu_automation import (
    MenuActionKind,
    MenuControllerStatus,
    MenuScene,
    MenuTransition,
)
from delta_vision.menu_worker import MenuLoopResult
from delta_vision.navigation import NavigationPolicy, NavigationStatus, RouteAction
from delta_vision.planner import RouteEdge, RouteNode
from delta_vision.template_profile import TemplateProfile
from delta_vision.worker import ColorAnchorSettings, ControlLoopResult, WorkerSettings

DEFAULT_TRANSITIONS = (
    MenuTransition(
        source=MenuScene.LOBBY,
        target=MenuScene.IN_MATCH,
        action_kind=MenuActionKind.CLICK,
    ),
)


@dataclass(frozen=True)
class _MenuProfile:
    frame_size: tuple[int, int] = (80, 60)
    profile_id: str = "menu-profile"
    transitions: tuple[MenuTransition, ...] = DEFAULT_TRANSITIONS
    observer: object = object()

    def create_controller(self) -> object:
        return object()


class _Source:
    def __init__(self) -> None:
        self.closed = False

    def grab(self):
        return None

    def close(self) -> None:
        self.closed = True


def _worker_settings(*, armed_ready: bool = True) -> WorkerSettings:
    graph = {
        "start": RouteNode(
            x=0,
            y=0,
            edges=(RouteEdge(target_node_id="goal", cost=1),),
        ),
        "goal": RouteNode(x=1, y=0, edges=()),
    }
    return WorkerSettings(
        target_window_title="三角洲行动",
        capture_backend="dxcam",
        armed_ready=armed_ready,
        emergency_virtual_key=123,
        max_key_hold_ms=250,
        loop_interval_ms=20,
        max_duration_seconds=30,
        perception=TemplateProfile(
            observer=SimpleNamespace(),
            frame_size=(80, 60),
            manifest_sha256="a" * 64,
            source_run_ids=frozenset({"route-cal"}),
            source_frame_sha256s=frozenset({"b" * 64}),
            source_perception_sha256s=frozenset({"c" * 64}),
            perception_regions=(),
        ),
        graph=graph,
        goal_node_id="goal",
        policy=NavigationPolicy(
            edge_actions={
                ("start", "goal"): RouteAction(key="w", mouse_dx=0, mouse_dy=0)
            },
            pulse_ms=80,
            min_progress_px=1,
            stuck_after_ms=500,
            localization_timeout_ms=500,
            max_recovery_attempts=0,
            recovery_keys=(),
            arrival_confirmations=1,
            initial_waypoint_confirmations=3,
            waypoint_advance_confirmations=2,
            relocalization_confirmations=3,
        ),
    )


def _settings(*, transitions=None, armed_ready: bool = True) -> GameSessionSettings:
    profile = _MenuProfile(
        transitions=DEFAULT_TRANSITIONS if transitions is None else transitions
    )
    return GameSessionSettings(
        worker=_worker_settings(armed_ready=armed_ready),
        menu_profile=profile,
        menu_loop_interval_ms=20,
        menu_max_duration_seconds=120,
    )


def test_load_session_settings_resolves_relative_menu_profile(tmp_path) -> None:
    config_path = tmp_path / "configs" / "game-route.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "menu": {
                    "profile": "../profiles/menu.json",
                    "loop_interval_ms": 30,
                    "max_duration_seconds": 90,
                }
            }
        ),
        encoding="utf-8",
    )
    loaded_paths: list[Path] = []

    settings = load_game_session_settings(
        config_path,
        worker_loader=lambda _path: _worker_settings(),
        menu_profile_loader=lambda path: loaded_paths.append(Path(path)) or _MenuProfile(),
    )

    assert loaded_paths == [(tmp_path / "profiles" / "menu.json").resolve()]
    assert settings.menu_loop_interval_ms == 30
    assert settings.menu_max_duration_seconds == 90


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("/tmp/menu.json", "相对路径"),
        ("C:\\menu.json", "相对路径"),
        ("", "非空相对路径"),
    ],
)
def test_load_session_settings_rejects_unsafe_menu_profile_reference(
    tmp_path, reference, message
) -> None:
    config_path = tmp_path / "game-route.json"
    config_path.write_text(
        json.dumps(
            {
                "menu": {
                    "profile": reference,
                    "loop_interval_ms": 20,
                    "max_duration_seconds": 90,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_game_session_settings(
            config_path,
            worker_loader=lambda _path: _worker_settings(),
            menu_profile_loader=lambda _path: _MenuProfile(),
        )


@pytest.mark.parametrize(
    "transitions",
    [
        (
            MenuTransition(
                MenuScene.STRATEGY_BOARD,
                MenuScene.IN_MATCH,
                MenuActionKind.CLICK,
            ),
        ),
        (
            MenuTransition(
                MenuScene.LOBBY,
                MenuScene.ZERO_DAM_READY,
                MenuActionKind.CLICK,
            ),
        ),
    ],
)
def test_session_settings_requires_lobby_to_in_match_flow(transitions) -> None:
    with pytest.raises(ValueError, match=r"LOBBY.*IN_MATCH"):
        _settings(transitions=transitions)


def test_session_settings_rejects_unsupported_menu_key() -> None:
    transitions = (
        MenuTransition(
            MenuScene.LOBBY,
            MenuScene.IN_MATCH,
            MenuActionKind.KEY,
            key="enter",
        ),
    )

    with pytest.raises(ValueError, match="不支持的菜单按键"):
        _settings(transitions=transitions)


def test_session_settings_rejects_schema_v1_color_anchor_route() -> None:
    worker = replace(
        _worker_settings(),
        perception=ColorAnchorSettings(
            bgr=(0, 255, 0),
            tolerance=5,
            minimum_area=10,
            confidence_threshold=0.9,
            localization_radius=20,
        ),
    )

    with pytest.raises(ValueError, match=r"schema_version=2.*TemplateProfile"):
        GameSessionSettings(
            worker=worker,
            menu_profile=_MenuProfile(),
            menu_loop_interval_ms=20,
            menu_max_duration_seconds=120,
        )


def test_session_settings_rejects_menu_and_route_resolution_mismatch() -> None:
    worker = replace(
        _worker_settings(),
        perception=replace(_worker_settings().perception, frame_size=(100, 100)),
    )

    with pytest.raises(ValueError, match="期望分辨率必须一致"):
        GameSessionSettings(
            worker=worker,
            menu_profile=_MenuProfile(),
            menu_loop_interval_ms=20,
            menu_max_duration_seconds=120,
        )


def test_windows_menu_runtime_rejects_resolution_before_opening_capture(tmp_path) -> None:
    source_calls = []

    with pytest.raises(ValueError, match="菜单 Profile 分辨率"):
        build_windows_menu_runtime(
            _settings(),
            artifacts=tmp_path,
            armed=False,
            region_resolver=lambda _title: CaptureRegion(0, 0, 100, 100),
            window_handle_resolver=lambda _title: 7,
            dxcam_factory=lambda region: source_calls.append(region) or _Source(),
        )

    assert source_calls == []


def test_armed_session_is_blocked_before_any_runner_when_not_ready(tmp_path) -> None:
    calls = []

    with pytest.raises(ValueError, match="armed_ready"):
        run_game_session(
            _settings(armed_ready=False),
            artifacts=tmp_path,
            armed=True,
            run_id="blocked-run",
            menu_runner=lambda **kwargs: calls.append(kwargs),
            route_runner=lambda *_args, **kwargs: calls.append(kwargs),
        )

    assert calls == []


def test_session_never_starts_route_when_menu_did_not_complete(tmp_path) -> None:
    route_calls = []

    result = run_game_session(
        _settings(),
        artifacts=tmp_path,
        armed=True,
        run_id="shared-run",
        menu_runner=lambda **_kwargs: MenuLoopResult(
            status=MenuControllerStatus.STOPPED,
            frame_count=4,
            action_count=1,
            duration_ns=20,
            reason="页面不确定",
        ),
        route_runner=lambda *_args, **kwargs: route_calls.append(kwargs),
    )

    assert result.status is GameSessionStatus.MENU_STOPPED
    assert result.route is None
    assert route_calls == []
    summary = json.loads((tmp_path / "session-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "menu_stopped"
    assert summary["run_id"] == "shared-run"
    assert summary["route"] is None


def test_session_starts_route_once_after_confirmed_in_match(tmp_path) -> None:
    calls = []

    def menu_runner(**kwargs):
        calls.append(("menu", kwargs))
        return MenuLoopResult(
            status=MenuControllerStatus.COMPLETED,
            frame_count=6,
            action_count=1,
            duration_ns=30,
            reason=None,
        )

    def route_runner(_worker, **kwargs):
        calls.append(("route", kwargs))
        return ControlLoopResult(
            status=NavigationStatus.ARRIVED,
            frame_count=8,
            duration_ns=40,
            reason=None,
        )

    result = run_game_session(
        _settings(),
        artifacts=tmp_path,
        armed=True,
        run_id="shared-run",
        menu_runner=menu_runner,
        route_runner=route_runner,
    )

    assert result.status is GameSessionStatus.COMPLETED
    assert [name for name, _kwargs in calls] == ["menu", "route"]
    assert calls[0][1]["artifacts"] == tmp_path / "menu"
    assert calls[1][1]["artifacts"] == tmp_path / "route"
    assert calls[0][1]["run_id"] == calls[1][1]["run_id"] == "shared-run"
    summary = json.loads((tmp_path / "session-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["route"]["status"] == "arrived"


def test_route_exception_keeps_menu_evidence_and_writes_failed_summary(tmp_path) -> None:
    menu_result = MenuLoopResult(
        status=MenuControllerStatus.COMPLETED,
        frame_count=6,
        action_count=1,
        duration_ns=30,
        reason=None,
    )

    with pytest.raises(RuntimeError, match="route failed"):
        run_game_session(
            _settings(),
            artifacts=tmp_path,
            armed=False,
            run_id="failed-run",
            menu_runner=lambda **_kwargs: menu_result,
            route_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("route failed")
            ),
        )

    summary = json.loads((tmp_path / "session-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failed_phase"] == "route"
    assert summary["menu"]["status"] == "completed"
    assert summary["route"] is None
    assert summary["error"]["exception_type"] == "RuntimeError"
