from dataclasses import FrozenInstanceError

import pytest

from delta_vision.config import CaptureRegion, RunnerConfig


def test_capture_region_exposes_dxcam_bounds() -> None:
    region = CaptureRegion(left=10, top=20, width=300, height=200)

    assert region.right == 310
    assert region.bottom == 220
    assert region.as_dxcam() == (10, 20, 310, 220)


@pytest.mark.parametrize(
    ("field", "value"),
    [("width", 0), ("height", -1)],
)
def test_capture_region_rejects_non_positive_size(field: str, value: int) -> None:
    values = {"left": 0, "top": 0, "width": 100, "height": 100, field: value}

    with pytest.raises(ValueError, match="宽高必须为正数"):
        CaptureRegion(**values)


def test_runner_config_is_immutable_and_safe_by_default() -> None:
    config = RunnerConfig(target_window_title="Delta Vision Test Target")

    assert config.armed is False
    assert config.max_key_hold_ms == 250
    assert config.confidence_threshold == pytest.approx(0.9)
    with pytest.raises(FrozenInstanceError):
        config.armed = True  # type: ignore[misc]


@pytest.mark.parametrize("threshold", [0, -0.01, 1.01])
def test_runner_config_rejects_invalid_confidence(threshold: float) -> None:
    with pytest.raises(ValueError, match="置信度阈值"):
        RunnerConfig(
            target_window_title="Delta Vision Test Target",
            confidence_threshold=threshold,
        )


def test_runner_config_accepts_full_confidence_threshold() -> None:
    config = RunnerConfig(
        target_window_title="Delta Vision Test Target",
        confidence_threshold=1,
    )

    assert config.confidence_threshold == 1


@pytest.mark.parametrize("value", [0, -1, True])
def test_runner_config_rejects_invalid_max_key_hold(value: int) -> None:
    with pytest.raises(ValueError, match="最大按键时长"):
        RunnerConfig(
            target_window_title="Delta Vision Test Target",
            max_key_hold_ms=value,
        )


def test_runner_config_rejects_armed_mode_without_window_guard() -> None:
    with pytest.raises(ValueError, match="目标窗口标题"):
        RunnerConfig(target_window_title="", armed=True)


def test_runner_config_rejects_non_boolean_armed_value() -> None:
    with pytest.raises(ValueError, match="armed 必须是布尔值"):
        RunnerConfig(
            target_window_title="Delta Vision Test Target",
            armed="false",  # type: ignore[arg-type]
        )
