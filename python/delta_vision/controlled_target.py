"""用于 Windows 黑盒 E2E 的独立可视化测试窗口。"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol

from .win32_native import ImeDisabledSession, enable_per_monitor_dpi_awareness

WINDOW_TITLE = "Delta Vision Test Target"
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
START_POSITION = (80.0, 520.0)
TURN_POSITION = (80.0, 80.0)
GOAL_POSITION = (700.0, 80.0)
GOAL_RADIUS = 20.0


class _Window(Protocol):
    report_callback_exception: Callable[
        [type[BaseException], BaseException, TracebackType | None], None
    ]

    def mainloop(self) -> None: ...

    def destroy(self) -> None: ...


class _ImeSession(Protocol):
    def disable(self) -> None: ...

    def restore(self) -> None: ...


class ControlledWindowLifetime:
    """把 IME 隔离与窗口主循环绑定，确保失败时不会发布可用状态。"""

    def __init__(self, window: _Window, ime_session: _ImeSession) -> None:
        self._window = window
        self._ime_session = ime_session
        self._started = False
        self._closed = False
        self._cleanup_error: BaseException | None = None
        self._callback_error: BaseException | None = None

    def _report_callback_exception(
        self,
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type
        if self._callback_error is None:
            self._callback_error = exception.with_traceback(traceback)
        self.close()

    def start(self, on_ready: Callable[[], None]) -> None:
        if self._started or self._closed:
            raise RuntimeError("受控窗口生命周期不能重复启动")
        try:
            # Tkinter 默认只打印回调异常；集中接管后才能恢复 IME 并让进程失败退出。
            self._window.report_callback_exception = self._report_callback_exception
            self._ime_session.disable()
        except BaseException:
            self._closed = True
            self._window.destroy()
            raise
        self._started = True
        try:
            on_ready()
        except BaseException as error:
            self.close()
            if self._cleanup_error is not None:
                raise error from self._cleanup_error
            raise

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._started:
                self._ime_session.restore()
        except BaseException as error:
            self._cleanup_error = self._cleanup_error or error
        finally:
            try:
                self._window.destroy()
            except BaseException as error:
                self._cleanup_error = self._cleanup_error or error
            self._closed = True

    def run(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("受控窗口必须成功启动后才能进入主循环")
        mainloop_error: BaseException | None = None
        try:
            self._window.mainloop()
        except BaseException as error:
            mainloop_error = error
        finally:
            self.close()
        primary_error = mainloop_error or self._callback_error
        if primary_error is not None:
            if self._cleanup_error is not None:
                raise primary_error from self._cleanup_error
            raise primary_error
        if self._cleanup_error is not None:
            raise self._cleanup_error


@dataclass(frozen=True, slots=True)
class TargetState:
    x: float
    y: float
    arrived: bool


class ControlledTargetModel:
    """不依赖 GUI 的确定性 WASD 运动模型。"""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        marker_radius: float,
        goal_radius: float,
        speed_px_per_second: float,
        ignore_input_ms: int = 0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("画布宽高必须为正数")
        if marker_radius <= 0 or goal_radius <= 0:
            raise ValueError("标记和目标半径必须为正数")
        if not math.isfinite(speed_px_per_second) or speed_px_per_second <= 0:
            raise ValueError("移动速度必须是正有限数")
        if ignore_input_ms < 0:
            raise ValueError("忽略输入时长不能为负数")
        self._width = width
        self._height = height
        self._goal = goal
        self._marker_radius = marker_radius
        self._goal_radius = goal_radius
        self._speed = speed_px_per_second
        self._ignore_input_ms = ignore_input_ms
        self._state = TargetState(float(start[0]), float(start[1]), False)

    @property
    def state(self) -> TargetState:
        return self._state

    def step(
        self,
        held_keys: frozenset[str],
        *,
        delta_seconds: float,
        elapsed_ms: int,
    ) -> TargetState:
        if self._state.arrived or elapsed_ms < self._ignore_input_ms:
            return self._state
        horizontal = int("d" in held_keys) - int("a" in held_keys)
        vertical = int("s" in held_keys) - int("w" in held_keys)
        distance = self._speed * max(0, delta_seconds)
        x = min(
            self._width - self._marker_radius,
            max(self._marker_radius, self._state.x + horizontal * distance),
        )
        y = min(
            self._height - self._marker_radius,
            max(self._marker_radius, self._state.y + vertical * distance),
        )
        arrived = (
            math.hypot(x - self._goal[0], y - self._goal[1])
            <= self._goal_radius
        )
        self._state = TargetState(x, y, arrived)
        return self._state


class GroundTruthWriter:
    """供独立评估器读取；Worker 不得读取此文件。"""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id 必须是非空字符串")
        self._path = Path(path)
        self._run_id = run_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # 每次目标进程都是独立试次，先清空旧成功记录，避免复用 artifacts 时误判。
        self._path.write_text("", encoding="utf-8")

    def write(self, event: str, *, at_ns: int, payload: Mapping[str, object]) -> None:
        record = {
            "event": event,
            "at_ns": at_ns,
            "payload": dict(payload),
            "schema_version": 1,
            "run_id": self._run_id,
        }
        serialized = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.write("\n")


def run_window(*, artifacts: Path, ignore_input_ms: int, run_id: str) -> int:
    """启动独立 Tk 窗口；模块层不导入 Tk，保证无桌面环境也能跑单测。"""

    enable_per_monitor_dpi_awareness()
    import tkinter as tk

    artifacts.mkdir(parents=True, exist_ok=True)
    writer = GroundTruthWriter(
        artifacts / "target-ground-truth.jsonl",
        run_id=run_id,
    )
    model = ControlledTargetModel(
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        start=START_POSITION,
        goal=GOAL_POSITION,
        marker_radius=10,
        goal_radius=GOAL_RADIUS,
        speed_px_per_second=420,
        ignore_input_ms=ignore_input_ms,
    )
    window = tk.Tk()
    window.title(WINDOW_TITLE)
    window.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}")
    window.resizable(False, False)
    canvas = tk.Canvas(
        window,
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        bg="#10151c",
        highlightthickness=0,
    )
    canvas.pack(fill="both", expand=True)
    window.update_idletasks()
    lifetime = ControlledWindowLifetime(
        window,
        ImeDisabledSession((int(window.winfo_id()), int(canvas.winfo_id()))),
    )
    canvas.create_line(
        START_POSITION[0],
        START_POSITION[1],
        TURN_POSITION[0],
        TURN_POSITION[1],
        GOAL_POSITION[0],
        GOAL_POSITION[1],
        fill="#405164",
        width=5,
    )
    goal = canvas.create_oval(
        GOAL_POSITION[0] - GOAL_RADIUS,
        GOAL_POSITION[1] - GOAL_RADIUS,
        GOAL_POSITION[0] + GOAL_RADIUS,
        GOAL_POSITION[1] + GOAL_RADIUS,
        fill="#ffd400",
        outline="",
    )
    marker = canvas.create_oval(0, 0, 0, 0, fill="#00ff00", outline="")
    status = canvas.create_text(
        CANVAS_WIDTH / 2,
        28,
        text="外部视觉 E2E 测试窗口 | F12 急停",
        fill="#dbe8f5",
        font=("Segoe UI", 15, "bold"),
    )
    held_keys: set[str] = set()
    started_at_ns = time.monotonic_ns()
    previous_tick_ns = started_at_ns
    last_logged_state = model.state

    def publish_start() -> None:
        writer.write(
            "start",
            at_ns=started_at_ns,
            payload={
                "x": model.state.x,
                "y": model.state.y,
                "ignore_input_ms": ignore_input_ms,
            },
        )

    def log_key(event_name: str, key: str) -> None:
        writer.write(event_name, at_ns=time.monotonic_ns(), payload={"key": key})

    def on_key_down(event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in {"w", "a", "s", "d"} and key not in held_keys:
            held_keys.add(key)
            log_key("key_down", key)

    def on_key_up(event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in held_keys:
            held_keys.remove(key)
            log_key("key_up", key)

    def on_focus_out(_: tk.Event) -> None:
        if held_keys:
            held_keys.clear()
            writer.write("focus_lost", at_ns=time.monotonic_ns(), payload={})

    def tick() -> None:
        nonlocal previous_tick_ns, last_logged_state
        now_ns = time.monotonic_ns()
        state = model.step(
            frozenset(held_keys),
            delta_seconds=(now_ns - previous_tick_ns) / 1_000_000_000,
            elapsed_ms=(now_ns - started_at_ns) // 1_000_000,
        )
        previous_tick_ns = now_ns
        canvas.coords(marker, state.x - 10, state.y - 10, state.x + 10, state.y + 10)
        if state != last_logged_state:
            writer.write(
                "position",
                at_ns=now_ns,
                payload={"x": state.x, "y": state.y, "arrived": state.arrived},
            )
            last_logged_state = state
        if state.arrived:
            canvas.itemconfigure(goal, fill="#48d597")
            canvas.itemconfigure(status, text="ARRIVED | 外部截图应连续确认目标")
        window.after(16, tick)

    def close_window() -> None:
        try:
            writer.write(
                "close",
                at_ns=time.monotonic_ns(),
                payload={
                    "arrived": model.state.arrived,
                    "x": model.state.x,
                    "y": model.state.y,
                },
            )
        finally:
            lifetime.close()

    window.bind("<KeyPress>", on_key_down)
    window.bind("<KeyRelease>", on_key_up)
    window.bind("<FocusOut>", on_focus_out)
    window.protocol("WM_DELETE_WINDOW", close_window)
    lifetime.start(publish_start)
    window.after(100, window.focus_force)
    window.after(16, tick)
    lifetime.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delta 外部视觉受控 E2E 测试窗口")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/controlled-target"),
        help="独立 ground truth 输出目录",
    )
    parser.add_argument("--ignore-input-ms", type=int, default=0)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    return run_window(
        artifacts=args.artifacts,
        ignore_input_ms=args.ignore_input_ms,
        run_id=args.run_id or uuid.uuid4().hex,
    )


if __name__ == "__main__":
    raise SystemExit(main())
