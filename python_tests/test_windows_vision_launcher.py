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
            '"SessionSample", "TestTarget", "Benchmark", "DryRun", "Armed", "SessionArmed", '
            '"LoopDryRun", "LoopArmed", "WarehouseDryRun", "WarehouseArmed", '
            '"ControlledE2E", "Preflight")]'
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
        "delta_vision.game_session",
        "delta_vision.external_session",
        "delta_vision.sample_frames",
        "delta_vision.session_sample",
        "delta_vision.calibrate_templates",
        '[ValidateSet("ncc", "orb", "sift")]',
        "--feature-backend $FeatureBackend",
        "--maximum-features $MaximumFeatures",
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

    session_block = script[
        script.index('if ($Mode -eq "SessionArmed")') : script.index(
            'if ($Mode -eq "Armed")'
        )
    ]
    assert "Enter-WorkerLock" in session_block
    assert "delta_vision.game_session" in session_block
    assert '"--armed"' in session_block
    assert "ReleaseMutex" in session_block
    assert script.index('if ($Mode -eq "SessionArmed")') < script.index(
        "New-Item -ItemType Directory -Path $Artifacts -Force"
    )

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
    assert "profiles\\route-01\\templates.json" not in script
    assert "choice /C DLAQ" in script
    assert "choice /C YN" in script
    assert "-Mode DryRun" in script
    assert "-Mode LoopDryRun" in script
    assert "-Mode LoopArmed" in script
    assert "-Mode SessionArmed" not in script
    assert "-ConfirmArmed" in script
    assert "F12" in script


def test_vision_external_modes_hold_mutex_and_preserve_python_exit_code() -> None:
    script = (PROJECT_ROOT / "vision.ps1").read_text(encoding="utf-8-sig")
    precreate_index = script.index(
        "New-Item -ItemType Directory -Path $Artifacts -Force"
    )

    for mode, next_mode in (
        ("LoopDryRun", "LoopArmed"),
        ("LoopArmed", "WarehouseDryRun"),
        ("WarehouseDryRun", "WarehouseArmed"),
        ("WarehouseArmed", "SessionArmed"),
    ):
        start = script.index(f'if ($Mode -eq "{mode}")')
        end = script.index(f'if ($Mode -eq "{next_mode}")')
        block = script[start:end]
        assert start < precreate_index
        assert "Enter-WorkerLock" in block
        assert "delta_vision.external_session" in block
        assert "--config $configPath" in block
        assert "--artifacts $Artifacts" in block
        assert "--run-id $effectiveRunId" in block
        assert "$workerExitCode = $LASTEXITCODE" in block
        assert "ReleaseMutex" in block
        assert "exit $workerExitCode" in block

    loop_dry = script[
        script.index('if ($Mode -eq "LoopDryRun")') : script.index(
            'if ($Mode -eq "LoopArmed")'
        )
    ]
    assert '"--armed"' not in loop_dry
    assert '"--confirm-armed"' not in loop_dry
    assert "Assert-ArmedConfirmation" not in loop_dry
    assert "--cleanup-only" not in loop_dry

    loop_armed = script[
        script.index('if ($Mode -eq "LoopArmed")') : script.index(
            'if ($Mode -eq "WarehouseDryRun")'
        )
    ]
    assert "Assert-ArmedConfirmation" in loop_armed
    assert '"--armed"' in loop_armed
    assert '"--confirm-armed"' in loop_armed
    assert "--cleanup-only" not in loop_armed

    warehouse_dry = script[
        script.index('if ($Mode -eq "WarehouseDryRun")') : script.index(
            'if ($Mode -eq "WarehouseArmed")'
        )
    ]
    assert "--cleanup-only" in warehouse_dry
    assert '"--armed"' not in warehouse_dry
    assert "Assert-ArmedConfirmation" not in warehouse_dry

    warehouse_armed = script[
        script.index('if ($Mode -eq "WarehouseArmed")') : script.index(
            'if ($Mode -eq "SessionArmed")'
        )
    ]
    assert "--cleanup-only" in warehouse_armed
    assert "Assert-ArmedConfirmation" in warehouse_armed
    assert '"--armed"' in warehouse_armed
    assert '"--confirm-armed"' in warehouse_armed


def test_warehouse_cleanup_cmd_defaults_to_dry_run_and_double_confirms_armed() -> None:
    script = (PROJECT_ROOT / "start-warehouse-cleanup.cmd").read_text(
        encoding="utf-8"
    )

    assert "configs\\game-route.json" in script
    assert "choice /C DAQ" in script
    assert "choice /C YN" in script
    assert "-Mode WarehouseDryRun" in script
    assert "-Mode WarehouseArmed" in script
    assert "-ConfirmArmed" in script
    assert "F12" in script
    assert "exit /b %EXIT_CODE%" in script


def test_route_calibration_cmd_enters_match_then_samples_without_route_profile() -> None:
    script = (PROJECT_ROOT / "start-route-calibration.cmd").read_text(
        encoding="utf-8"
    )

    assert "profiles\\menu-zero-cost\\menu.json" in script
    assert "choice /C YN" in script
    assert "-Mode SessionSample" in script
    assert "-ConfirmArmed" in script
    assert "-Split calibration" in script
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
    assert config["menu"] == {
        "profile": "../profiles/menu-zero-cost/menu.json",
        "loop_interval_ms": 20,
        "max_duration_seconds": 120,
    }
    assert config["external_loop"] == {
        "cycle_limit": 1,
        "return": {
            "confirmation_frames": 3,
            "maximum_confirmation_span_ms": 500,
            "maximum_frame_age_ms": 500,
            "loop_interval_ms": 20,
            "max_duration_seconds": 180,
        },
    }
    assert config["warehouse_cleanup"] == {
        "profile": "../profiles/warehouse-empty/menu.json",
        "armed_ready": False,
        "loop_interval_ms": 20,
        "max_duration_seconds": 30,
    }
    assert config["goal_node_id"] in config["nodes"]
    assert config["edge_actions"]
    assert all(type(action["mouse_dx"]) is int for action in config["edge_actions"])
    assert all(type(action["mouse_dy"]) is int for action in config["edge_actions"])
