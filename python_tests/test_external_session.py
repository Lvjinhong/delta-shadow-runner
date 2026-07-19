import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import delta_vision.external_session as session_module
from delta_vision.external_loop import (
    CleanupSessionResult,
    ExternalLoopSettings,
    ExternalLoopStatus,
)
from delta_vision.game_session import WindowsMenuSettings
from delta_vision.menu_automation import (
    MenuActionKind,
    MenuScene,
    MenuTransition,
)
from delta_vision.passive_scene import PassiveReturnPolicy
from delta_vision.warehouse_cleanup import (
    WAREHOUSE_BASE_TEMPLATE_ID,
    WAREHOUSE_EMPTY_EVIDENCE_IDS,
    WAREHOUSE_EMPTY_TEMPLATE_ID,
    WarehouseCleanupPolicy,
)
from delta_vision.warehouse_worker import (
    WindowsWarehouseCleanupSession,
    WindowsWarehouseCleanupSettings,
)


def _entry_profile() -> SimpleNamespace:
    return SimpleNamespace(
        profile_id="entry-1080p",
        frame_size=(1920, 1080),
        transitions=(
            MenuTransition(
                source=MenuScene.BASE,
                target=MenuScene.IN_MATCH,
                action_kind=MenuActionKind.CLICK,
            ),
        ),
    )


def _warehouse_profile(*, frame_size: tuple[int, int] = (1920, 1080)) -> SimpleNamespace:
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
        profile_id="warehouse-empty-1080p",
        frame_size=frame_size,
        observer=SimpleNamespace(observe=lambda _frame: None),
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
        confirmation_frames=4,
        maximum_confirmation_span_ms=640,
        maximum_frame_age_ms=240,
        transition_timeout_ms=7100,
        maximum_action_point_drift_px=7.0,
    )


def _match_settings(*, armed_ready: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        armed_ready=armed_ready,
        target_window_title="三角洲行动  ",
        capture_backend="mss",
        emergency_virtual_key=123,
        max_key_hold_ms=250,
        perception=SimpleNamespace(frame_size=(1920, 1080)),
    )


def _game_settings(
    *,
    armed_ready: bool = True,
    match_backend: str = "mss",
    match_max_key_hold_ms: int = 250,
) -> SimpleNamespace:
    profile = _entry_profile()
    match = _match_settings(armed_ready=armed_ready)
    match.capture_backend = match_backend
    match.max_key_hold_ms = match_max_key_hold_ms
    entry = WindowsMenuSettings(
        target_window_title=match.target_window_title,
        capture_backend="mss",
        emergency_virtual_key=match.emergency_virtual_key,
        max_key_hold_ms=250,
        menu_profile=profile,
        loop_interval_ms=20,
        max_duration_seconds=120,
    )
    return SimpleNamespace(
        worker=match,
        menu_profile=profile,
        menu_runtime_settings=lambda: entry,
    )


def _config(tmp_path: Path, *, warehouse_ready: bool = True) -> Path:
    config_path = tmp_path / "configs" / "game-route.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "target_window_title": "三角洲行动  ",
                "capture_backend": "mss",
                "emergency_virtual_key": 123,
                "max_key_hold_ms": 250,
                "external_loop": {
                    "cycle_limit": 2,
                    "return": {
                        "confirmation_frames": 4,
                        "maximum_confirmation_span_ms": 600,
                        "maximum_frame_age_ms": 400,
                        "loop_interval_ms": 25,
                        "max_duration_seconds": 170,
                    },
                },
                "warehouse_cleanup": {
                    "profile": "../profiles/warehouse/menu.json",
                    "armed_ready": warehouse_ready,
                    "loop_interval_ms": 30,
                    "max_duration_seconds": 20,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_external_session_composes_loop_and_cleanup_settings(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    warehouse_profile = _warehouse_profile()
    loaded_paths: list[Path] = []

    settings = session_module.load_external_session_settings(
        config_path,
        game_session_loader=lambda _path: _game_settings(),
        menu_profile_loader=lambda path: (
            loaded_paths.append(Path(path)) or warehouse_profile
        ),
    )

    assert isinstance(settings.external_loop, ExternalLoopSettings)
    assert settings.external_loop.cycle_limit == 2
    assert settings.external_loop.return_policy == PassiveReturnPolicy(
        confirmation_frames=4,
        maximum_confirmation_span_ms=600,
        maximum_frame_age_ms=400,
        loop_interval_ms=25,
        max_duration_seconds=170,
    )
    cleanup = settings.warehouse_cleanup
    assert isinstance(cleanup, WindowsWarehouseCleanupSettings)
    assert cleanup.target_window_title == "三角洲行动  "
    assert cleanup.capture_backend == "mss"
    assert cleanup.emergency_virtual_key == 123
    assert cleanup.armed_ready is True
    assert cleanup.policy == WarehouseCleanupPolicy(
        confirmation_frames=4,
        maximum_confirmation_span_ms=640,
        maximum_frame_age_ms=240,
        transition_timeout_ms=7100,
        maximum_action_point_drift_px=7.0,
    )
    assert loaded_paths == [(tmp_path / "profiles/warehouse/menu.json").resolve()]


@pytest.mark.parametrize(
    "reference",
    [
        "/tmp/warehouse/menu.json",
        "C:\\profiles\\warehouse\\menu.json",
        "../../outside/menu.json",
    ],
)
def test_cleanup_profile_must_be_project_relative(
    tmp_path: Path,
    reference: str,
) -> None:
    config_path = _config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["warehouse_cleanup"]["profile"] = reference
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"仓库.*Profile.*项目内相对路径"):
        session_module.load_warehouse_cleanup_settings(
            config_path,
            menu_profile_loader=lambda _path: pytest.fail("不应加载越界 Profile"),
        )


def test_warehouse_policy_rejects_non_1080_profile_before_runtime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1920x1080"):
        session_module.load_warehouse_cleanup_settings(
            _config(tmp_path),
            menu_profile_loader=lambda _path: _warehouse_profile(
                frame_size=(2560, 1440)
            ),
        )


@pytest.mark.parametrize(
    ("route_ready", "warehouse_ready", "message"),
    [
        (False, True, "路线 armed_ready=false"),
        (True, False, "仓库 armed_ready=false"),
    ],
)
def test_external_armed_requires_both_readiness_flags_before_runner(
    tmp_path: Path,
    route_ready: bool,
    warehouse_ready: bool,
    message: str,
) -> None:
    config_path = _config(tmp_path, warehouse_ready=warehouse_ready)
    settings = session_module.load_external_session_settings(
        config_path,
        game_session_loader=lambda _path: _game_settings(armed_ready=route_ready),
        menu_profile_loader=lambda _path: _warehouse_profile(),
    )
    artifacts = tmp_path / "artifacts"

    with pytest.raises(ValueError, match=message):
        session_module.run_windows_external_session(
            settings,
            artifacts=artifacts,
            armed=True,
            run_id="external-armed",
            external_runner=lambda *_args, **_kwargs: pytest.fail(
                "授权失败时不应启动外循环"
            ),
        )

    assert not artifacts.exists()


def test_external_session_injects_concrete_cleanup_session(tmp_path: Path) -> None:
    settings = session_module.load_external_session_settings(
        _config(tmp_path),
        game_session_loader=lambda _path: _game_settings(),
        menu_profile_loader=lambda _path: _warehouse_profile(),
    )
    captured: dict[str, object] = {}
    expected = SimpleNamespace(status=ExternalLoopStatus.COMPLETED)

    def external_runner(loop_settings: object, **kwargs: object) -> object:
        captured["settings"] = loop_settings
        captured.update(kwargs)
        return expected

    result = session_module.run_windows_external_session(
        settings,
        artifacts=tmp_path / "loop",
        armed=False,
        run_id="external-dry",
        external_runner=external_runner,
    )

    assert result is expected
    assert captured["settings"] is settings.external_loop
    assert captured["armed"] is False
    assert captured["run_id"] == "external-dry"
    assert isinstance(captured["cleanup_session"], WindowsWarehouseCleanupSession)


@pytest.mark.parametrize(
    ("game_settings", "message"),
    [
        (_game_settings(match_backend="dxcam"), "截图后端必须一致"),
        (_game_settings(match_max_key_hold_ms=999), "最大按键保持时长必须一致"),
    ],
)
def test_external_session_rejects_match_runtime_parameter_drift(
    tmp_path: Path,
    game_settings: SimpleNamespace,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        session_module.load_external_session_settings(
            _config(tmp_path),
            game_session_loader=lambda _path: game_settings,
            menu_profile_loader=lambda _path: _warehouse_profile(),
        )


def test_main_validate_only_has_no_runtime_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _config(tmp_path)
    settings = session_module.load_external_session_settings(
        config_path,
        game_session_loader=lambda _path: _game_settings(),
        menu_profile_loader=lambda _path: _warehouse_profile(),
    )
    monkeypatch.setattr(
        session_module,
        "load_external_session_settings",
        lambda _path: settings,
    )
    monkeypatch.setattr(
        session_module,
        "run_windows_external_session",
        lambda *_args, **_kwargs: pytest.fail("validate-only 不应启动外循环"),
    )
    artifacts = tmp_path / "runtime"

    exit_code = session_module.main(
        [
            "--config",
            str(config_path),
            "--artifacts",
            str(artifacts),
            "--validate-only",
        ]
    )

    assert exit_code == 0
    assert not artifacts.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid"
    assert payload["entry_profile_id"] == "entry-1080p"
    assert payload["warehouse_profile_id"] == "warehouse-empty-1080p"
    assert payload["route_armed_ready"] is True
    assert payload["warehouse_armed_ready"] is True


def test_main_rejects_armed_without_independent_cli_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        session_module,
        "load_external_session_settings",
        lambda _path: pytest.fail("缺少确认时不应读取配置或创建运行时"),
    )

    exit_code = session_module.main(
        ["--config", str(_config(tmp_path)), "--armed"]
    )

    assert exit_code == 1
    assert "--confirm-armed" in capsys.readouterr().err


def test_cleanup_only_validate_has_no_session_or_artifact_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cleanup_settings = WindowsWarehouseCleanupSettings(
        target_window_title="三角洲行动  ",
        capture_backend="mss",
        emergency_virtual_key=123,
        max_key_hold_ms=250,
        menu_profile=_warehouse_profile(),
        loop_interval_ms=20,
        max_duration_seconds=20,
        armed_ready=False,
        policy=WarehouseCleanupPolicy(
            confirmation_frames=4,
            maximum_confirmation_span_ms=640,
            maximum_frame_age_ms=240,
            transition_timeout_ms=7100,
            maximum_action_point_drift_px=7.0,
        ),
    )
    monkeypatch.setattr(
        session_module,
        "load_warehouse_cleanup_settings",
        lambda _path: cleanup_settings,
    )
    monkeypatch.setattr(
        session_module,
        "WindowsWarehouseCleanupSession",
        lambda **_kwargs: pytest.fail("validate-only 不应构造仓库 session"),
    )
    artifacts = tmp_path / "cleanup"

    exit_code = session_module.main(
        [
            "--config",
            str(_config(tmp_path)),
            "--artifacts",
            str(artifacts),
            "--cleanup-only",
            "--validate-only",
        ]
    )

    assert exit_code == 0
    assert not artifacts.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "valid",
        "mode": "warehouse_cleanup",
        "config": str(tmp_path / "configs/game-route.json"),
        "target_window_title": "三角洲行动  ",
        "warehouse_profile_id": "warehouse-empty-1080p",
        "warehouse_frame_size": [1920, 1080],
        "warehouse_armed_ready": False,
    }


@pytest.mark.parametrize(("completed", "expected_exit"), [(True, 0), (False, 2)])
def test_cleanup_only_main_forwards_armed_and_preserves_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    completed: bool,
    expected_exit: int,
) -> None:
    cleanup_settings = SimpleNamespace(profile_id="cleanup-settings")
    calls: dict[str, object] = {}

    class FakeCleanupSession:
        def __init__(self, *, settings: object) -> None:
            calls["settings"] = settings

        def run(self, **kwargs: object) -> CleanupSessionResult:
            calls.update(kwargs)
            return CleanupSessionResult(
                completed=completed,
                reason=None if completed else "安全停止",
                summary={
                    "status": "completed" if completed else "stopped",
                    "completed": completed,
                    "run_id": kwargs["run_id"],
                },
            )

    monkeypatch.setattr(
        session_module,
        "load_warehouse_cleanup_settings",
        lambda _path: cleanup_settings,
    )
    monkeypatch.setattr(
        session_module,
        "WindowsWarehouseCleanupSession",
        FakeCleanupSession,
    )
    artifacts = tmp_path / f"cleanup-{completed}"

    exit_code = session_module.main(
        [
            "--config",
            str(_config(tmp_path)),
            "--artifacts",
            str(artifacts),
            "--run-id",
            "cleanup-cli",
            "--cleanup-only",
            "--armed",
            "--confirm-armed",
        ]
    )

    assert exit_code == expected_exit
    assert calls == {
        "settings": cleanup_settings,
        "artifacts": artifacts,
        "armed": True,
        "run_id": "cleanup-cli",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is completed
    assert payload["artifacts"] == str(artifacts)


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        (ExternalLoopStatus.COMPLETED, 0),
        (ExternalLoopStatus.STOPPED, 2),
    ],
)
def test_external_main_preserves_result_exit_code_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: ExternalLoopStatus,
    expected_exit: int,
) -> None:
    settings = SimpleNamespace(name="settings")
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        session_module,
        "load_external_session_settings",
        lambda _path: settings,
    )

    def runner(loaded: object, **kwargs: object) -> object:
        calls["settings"] = loaded
        calls.update(kwargs)
        return SimpleNamespace(status=status, completed_cycles=1)

    monkeypatch.setattr(session_module, "run_windows_external_session", runner)
    artifacts = tmp_path / f"loop-{status}"

    exit_code = session_module.main(
        [
            "--config",
            str(_config(tmp_path)),
            "--artifacts",
            str(artifacts),
            "--run-id",
            "loop-cli",
        ]
    )

    assert exit_code == expected_exit
    assert calls == {
        "settings": settings,
        "artifacts": artifacts,
        "armed": False,
        "run_id": "loop-cli",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == str(status)
    assert payload["completed_cycles"] == 1
    assert payload["artifacts"] == str(artifacts)
    assert payload["run_id"] == "loop-cli"


def test_main_maps_configuration_error_to_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        session_module,
        "load_external_session_settings",
        lambda _path: (_ for _ in ()).throw(ValueError("配置损坏")),
    )

    exit_code = session_module.main(["--config", str(_config(tmp_path))])

    assert exit_code == 1
    assert "配置损坏" in capsys.readouterr().err


def test_duplicate_config_field_is_rejected_before_profile_load(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config_path.write_text(
        '{"schema_version":2,"schema_version":2}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复字段"):
        session_module.load_warehouse_cleanup_settings(
            config_path,
            menu_profile_loader=lambda _path: pytest.fail("不应加载 Profile"),
        )
