import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import delta_vision.session_sample as session_sample_module
from delta_vision.menu_automation import MenuControllerStatus, MenuScene
from delta_vision.menu_worker import MenuLoopResult
from delta_vision.sample_frames import SamplingResult
from delta_vision.session_sample import (
    SessionSamplingStatus,
    main,
    run_session_sampling,
)


@dataclass(frozen=True)
class _MenuSettings:
    target_window_title: str = "三角洲行动"
    capture_backend: str = "dxcam"


def _menu_result(status: MenuControllerStatus) -> MenuLoopResult:
    return MenuLoopResult(
        status=status,
        frame_count=6,
        action_count=2,
        duration_ns=30,
        reason=None if status is MenuControllerStatus.COMPLETED else "页面不确定",
        terminal_scene=(
            MenuScene.IN_MATCH
            if status is MenuControllerStatus.COMPLETED
            else MenuScene.UNKNOWN
        ),
    )


def _sampling_result(run_id: str, split: str) -> SamplingResult:
    return SamplingResult(
        run_id=run_id,
        dataset_split=split,
        window_title="三角洲行动",
        backend="dxcam",
        requested_duration_seconds=120,
        measured_duration_seconds=120,
        sample_fps=5,
        frame_count=600,
        no_frame_count=0,
        resolution=(2560, 1440),
    )


def test_session_sample_never_starts_sampling_when_menu_did_not_complete(
    tmp_path: Path,
) -> None:
    sampling_calls = []
    artifacts = tmp_path / "run"

    result = run_session_sampling(
        _MenuSettings(),
        artifacts=artifacts,
        run_id="route-cal",
        dataset_split="calibration",
        duration_seconds=120,
        sample_fps=5,
        menu_runner=lambda **_kwargs: _menu_result(MenuControllerStatus.STOPPED),
        sampling_runner=lambda **kwargs: sampling_calls.append(kwargs),
    )

    assert result.status is SessionSamplingStatus.MENU_STOPPED
    assert result.sampling is None
    assert sampling_calls == []
    summary = json.loads(
        (artifacts / "session-sample-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "menu_stopped"
    assert summary["sampling"] is None


def test_session_sample_starts_immediately_after_confirmed_in_match(
    tmp_path: Path,
) -> None:
    calls = []
    artifacts = tmp_path / "run"

    def menu_runner(**kwargs):
        calls.append(("menu", kwargs))
        return _menu_result(MenuControllerStatus.COMPLETED)

    def sampling_runner(**kwargs):
        calls.append(("sampling", kwargs))
        return _sampling_result(kwargs["run_id"], kwargs["dataset_split"])

    result = run_session_sampling(
        _MenuSettings(),
        artifacts=artifacts,
        run_id="route-cal",
        dataset_split="calibration",
        duration_seconds=120,
        sample_fps=5,
        menu_runner=menu_runner,
        sampling_runner=sampling_runner,
    )

    assert result.status is SessionSamplingStatus.COMPLETED
    assert [name for name, _kwargs in calls] == ["menu", "sampling"]
    assert calls[0][1]["run_id"] == calls[1][1]["run_id"] == "route-cal"
    assert calls[0][1]["artifacts"] == artifacts / "menu"
    assert calls[0][1]["armed"] is True
    assert calls[1][1]["output_directory"] == artifacts / "dataset"
    assert calls[1][1]["start_delay_seconds"] == 0
    summary = json.loads(
        (artifacts / "session-sample-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "completed"
    assert summary["sampling"]["frame_count"] == 600


def test_session_sample_rejects_completed_menu_without_in_match_scene(
    tmp_path: Path,
) -> None:
    sampling_calls = []
    artifacts = tmp_path / "run"
    wrong_terminal = MenuLoopResult(
        status=MenuControllerStatus.COMPLETED,
        frame_count=6,
        action_count=2,
        duration_ns=30,
        reason=None,
        terminal_scene=MenuScene.LOBBY,
    )

    with pytest.raises(RuntimeError, match="IN_MATCH"):
        run_session_sampling(
            _MenuSettings(),
            artifacts=artifacts,
            run_id="route-cal",
            dataset_split="calibration",
            duration_seconds=120,
            sample_fps=5,
            menu_runner=lambda **_kwargs: wrong_terminal,
            sampling_runner=lambda **kwargs: sampling_calls.append(kwargs),
        )

    assert sampling_calls == []
    summary = json.loads(
        (artifacts / "session-sample-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["failed_phase"] == "menu_contract"


def test_session_sample_rejects_invalid_run_id_before_menu(tmp_path: Path) -> None:
    menu_calls = []
    artifacts = tmp_path / "run"

    with pytest.raises(ValueError, match="run_id"):
        run_session_sampling(
            _MenuSettings(),
            artifacts=artifacts,
            run_id="../escape",
            dataset_split="calibration",
            duration_seconds=120,
            sample_fps=5,
            menu_runner=lambda **kwargs: menu_calls.append(kwargs),
            sampling_runner=lambda **_kwargs: None,
        )

    assert menu_calls == []
    assert not artifacts.exists()


def test_session_sample_rejects_existing_artifact_root_before_menu(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "existing"
    artifacts.mkdir()
    (artifacts / "owned.txt").write_text("keep", encoding="utf-8")
    menu_calls = []

    with pytest.raises(FileExistsError, match="运行目录已经存在"):
        run_session_sampling(
            _MenuSettings(),
            artifacts=artifacts,
            run_id="route-cal",
            dataset_split="calibration",
            duration_seconds=120,
            sample_fps=5,
            menu_runner=lambda **kwargs: menu_calls.append(kwargs),
            sampling_runner=lambda **_kwargs: None,
        )

    assert menu_calls == []
    assert (artifacts / "owned.txt").read_text(encoding="utf-8") == "keep"


def test_session_sample_preserves_menu_evidence_when_sampling_fails(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "run"

    def fail_sampling(**_kwargs):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        run_session_sampling(
            _MenuSettings(),
            artifacts=artifacts,
            run_id="route-cal",
            dataset_split="calibration",
            duration_seconds=120,
            sample_fps=5,
            menu_runner=lambda **_kwargs: _menu_result(
                MenuControllerStatus.COMPLETED
            ),
            sampling_runner=fail_sampling,
        )

    summary = json.loads(
        (artifacts / "session-sample-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["failed_phase"] == "sampling"
    assert summary["menu"]["status"] == "completed"
    assert summary["sampling"] is None


def test_session_sample_does_not_mask_sampling_error_when_failure_summary_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = RuntimeError("capture failed")

    def fail_sampling(**_kwargs):
        raise original

    def fail_summary(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(session_sample_module, "write_atomic_json", fail_summary)

    with pytest.raises(RuntimeError, match="capture failed") as raised:
        run_session_sampling(
            _MenuSettings(),
            artifacts=tmp_path / "run",
            run_id="route-cal",
            dataset_split="calibration",
            duration_seconds=120,
            sample_fps=5,
            menu_runner=lambda **_kwargs: _menu_result(
                MenuControllerStatus.COMPLETED
            ),
            sampling_runner=fail_sampling,
        )

    assert raised.value is original
    assert any("disk full" in note for note in raised.value.__notes__)


def test_session_sample_cli_requires_explicit_armed_confirmation(capsys) -> None:
    exit_code = main(
        [
            "--menu-profile",
            "menu.json",
            "--artifacts",
            "artifacts/run",
            "--split",
            "calibration",
        ]
    )

    assert exit_code == 1
    assert "--armed" in capsys.readouterr().err
