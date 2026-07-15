import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_vision_powershell_bootstrap_has_safe_reproducible_contract() -> None:
    path = PROJECT_ROOT / "vision.ps1"
    raw = path.read_bytes()
    script = raw.decode("utf-8-sig")

    assert raw.startswith(b"\xef\xbb\xbf")
    for fragment in (
        (
            '[ValidateSet("Setup", "Sample", "Calibrate", "Evaluate", '
            '"TestTarget", "Benchmark", "DryRun", "Armed", "ControlledE2E", '
            '"Preflight")]'
        ),
        "winget.exe install --id astral-sh.uv -e",
        "https://astral.sh/uv/0.11.28/install.ps1",
        "python install 3.12",
        "sync --frozen --python 3.12",
        "Local\\DeltaVisionWorker",
        "ConfirmArmed",
        "delta_vision.controlled_target",
        "delta_vision.benchmark",
        '"--duration", "60"',
        "delta_vision.worker",
        "delta_vision.sample_frames",
        "delta_vision.calibrate_templates",
        "delta_vision.evaluate_templates",
        "--validate-only",
        "delta_vision.preflight",
        "capture-gate.json",
        "preflight-report.json",
        "$effectiveRunId",
        "--run-id",
        "--config-exit-code",
        "--controlled-exit-code",
        "--benchmark-exit-code",
        "Invoke-ControlledE2E",
        '"--split"',
        '"--armed"',
        "taskkill.exe",
        "Wait-ControlledTargetArrival",
        "target-ground-truth.jsonl",
        '$event.event -eq "start"',
        '$event.event -eq "position"',
        "$sawStart = $true",
        "$event.payload.arrived -eq $true",
        "$workerExitCode = 3",
        "$taskkillExitCode = $LASTEXITCODE",
        ".WaitForExit(2000)",
        "$workerExitCode = 4",
    ):
        assert fragment in script, f"vision.ps1 缺少契约片段: {fragment}"

    dry_run_block = script[
        script.index('if ($Mode -eq "DryRun")') : script.index(
            'if ($Mode -eq "Armed")'
        )
    ]
    assert "Enter-WorkerLock" in dry_run_block
    assert "ReleaseMutex" in dry_run_block

    controlled_function = script[
        script.index("function Invoke-ControlledE2E") : script.index(
            "$uv = Initialize-PythonEnvironment"
        )
    ]
    assert "-ArgumentList $targetArgumentLine" in controlled_function
    assert '$targetArtifactsArgument = \'"\' + $targetArtifactsPath + \'"\'' in controlled_function
    assert "$targetArguments = @(" not in controlled_function


def test_controlled_e2e_cmd_requires_explicit_confirmation() -> None:
    script = (PROJECT_ROOT / "start-controlled-e2e.cmd").read_text(
        encoding="utf-8"
    )

    assert "choice /C YN" in script
    assert "-ExecutionPolicy Bypass" in script
    assert "-Mode ControlledE2E" in script
    assert "-ConfirmArmed" in script
    assert "start-demo.cmd" not in script


def test_game_route_cmd_defaults_to_dry_run_and_double_confirms_armed() -> None:
    script = (PROJECT_ROOT / "start-game-route.cmd").read_text(encoding="utf-8")

    assert "configs\\game-route.json" in script
    assert "configs\\game-route.example.json" in script
    assert "profiles\\route-01\\templates.json" in script
    assert "choice /C DAQ" in script
    assert "choice /C YN" in script
    assert "-Mode DryRun" in script
    assert "-Mode Armed" in script
    assert "-ConfirmArmed" in script
    assert "F12" in script


def test_windows_preflight_cmd_requires_game_config_and_explicit_confirmation() -> None:
    script = (PROJECT_ROOT / "start-windows-preflight.cmd").read_text(
        encoding="utf-8"
    )

    assert "configs\\game-route.json" in script
    assert "choice /C YN" in script
    assert "-Mode Preflight" in script
    assert "-ConfirmArmed" in script
    assert "60" in script
    assert "F12" in script


def test_game_route_example_is_safe_template_profile_config() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "game-route.example.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["schema_version"] == 2
    assert config["armed_ready"] is False
    assert config["target_window_title"] == "三角洲行动"
    assert config["perception"]["mode"] == "template"
    assert config["perception"]["template_profile"] == "../profiles/route-01/templates.json"
    assert config["goal_node_id"] in config["nodes"]
    assert config["edge_actions"]
    assert all(type(action["mouse_dx"]) is int for action in config["edge_actions"])
    assert all(type(action["mouse_dy"]) is int for action in config["edge_actions"])
