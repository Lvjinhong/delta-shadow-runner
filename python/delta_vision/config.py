"""Worker 的不可变运行配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """使用屏幕像素表示的采集区域。"""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("采集区域宽高必须为正数")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_dxcam(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """第一条纵向链路使用的安全配置。"""

    target_window_title: str
    armed: bool = False
    max_key_hold_ms: int = 250
    confidence_threshold: float = 0.9

    def __post_init__(self) -> None:
        if type(self.armed) is not bool:
            raise ValueError("armed 必须是布尔值")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("置信度阈值必须大于 0 且不超过 1")
        if isinstance(self.max_key_hold_ms, bool) or self.max_key_hold_ms <= 0:
            raise ValueError("最大按键时长必须为正数")
        if self.armed and not self.target_window_title.strip():
            raise ValueError("armed 模式必须配置目标窗口标题")
