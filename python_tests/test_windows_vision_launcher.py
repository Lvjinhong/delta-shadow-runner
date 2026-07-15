from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_vision_powershell_bootstrap_has_safe_reproducible_contract() -> None:
    path = PROJECT_ROOT / "vision.ps1"
    raw = path.read_bytes()
    script = raw.decode("utf-8-sig")

    assert raw.startswith(b"\xef\xbb\xbf")
    for fragment in (
        '[ValidateSet("Setup", "TestTarget", "DryRun", "Armed", "ControlledE2E")]',
        "winget.exe install --id astral-sh.uv -e",
        "https://astral.sh/uv/0.11.28/install.ps1",
        "python install 3.12",
        "sync --frozen --python 3.12",
        "Local\\DeltaVisionWorker",
        "ConfirmArmed",
        "delta_vision.controlled_target",
        "delta_vision.worker",
        '"--armed"',
        "taskkill.exe",
    ):
        assert fragment in script, f"vision.ps1 缺少契约片段: {fragment}"


def test_controlled_e2e_cmd_requires_explicit_confirmation() -> None:
    script = (PROJECT_ROOT / "start-controlled-e2e.cmd").read_text(
        encoding="utf-8"
    )

    assert "choice /C YN" in script
    assert "-ExecutionPolicy Bypass" in script
    assert "-Mode ControlledE2E" in script
    assert "-ConfirmArmed" in script
    assert "start-demo.cmd" not in script
