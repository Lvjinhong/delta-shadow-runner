import json
from pathlib import Path

import pytest

from delta_vision.controlled_target import (
    ControlledTargetModel,
    ControlledWindowLifetime,
    GroundTruthWriter,
)


class FakeImeSession:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.disable_error: Exception | None = None
        self.restore_error: Exception | None = None

    def disable(self) -> None:
        self.events.append("disable_ime")
        if self.disable_error is not None:
            raise self.disable_error

    def restore(self) -> None:
        self.events.append("restore_ime")
        if self.restore_error is not None:
            raise self.restore_error


class FakeWindow:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.mainloop_error: Exception | None = None
        self.on_mainloop = None
        self.report_callback_exception = self._default_report_callback_exception

    def _default_report_callback_exception(
        self, exception_type, exception, traceback
    ) -> None:
        self.events.append("tk_reported_callback_error")

    def mainloop(self) -> None:
        self.events.append("mainloop")
        if self.on_mainloop is not None:
            try:
                self.on_mainloop()
            except BaseException as error:
                self.report_callback_exception(
                    type(error), error, error.__traceback__
                )
        if self.mainloop_error is not None:
            raise self.mainloop_error

    def destroy(self) -> None:
        self.events.append("destroy")


def test_controlled_target_enables_dpi_awareness_before_importing_tkinter() -> None:
    source = (
        Path(__file__).parents[1]
        / "python"
        / "delta_vision"
        / "controlled_target.py"
    ).read_text(encoding="utf-8")

    assert source.index("enable_per_monitor_dpi_awareness()") < source.index(
        "import tkinter as tk"
    )


def test_controlled_window_lifetime_disables_ime_before_ready_and_mainloop() -> None:
    events: list[str] = []
    window = FakeWindow(events)
    lifetime = ControlledWindowLifetime(window, FakeImeSession(events))

    lifetime.start(lambda: events.append("ready"))
    lifetime.run()

    assert events == [
        "disable_ime",
        "ready",
        "mainloop",
        "restore_ime",
        "destroy",
    ]


def test_controlled_window_lifetime_fails_closed_before_ready() -> None:
    events: list[str] = []
    window = FakeWindow(events)
    ime_session = FakeImeSession(events)
    ime_session.disable_error = OSError("无法禁用 IME")
    lifetime = ControlledWindowLifetime(window, ime_session)

    with pytest.raises(OSError, match="无法禁用 IME"):
        lifetime.start(lambda: events.append("ready"))

    assert events == ["disable_ime", "destroy"]


def test_controlled_window_lifetime_restores_before_destroy_on_close() -> None:
    events: list[str] = []
    window = FakeWindow(events)
    lifetime = ControlledWindowLifetime(window, FakeImeSession(events))
    lifetime.start(lambda: events.append("ready"))

    lifetime.close()
    lifetime.close()

    assert events == ["disable_ime", "ready", "restore_ime", "destroy"]


def test_controlled_window_lifetime_cleans_up_when_mainloop_raises() -> None:
    events: list[str] = []
    window = FakeWindow(events)
    window.mainloop_error = RuntimeError("主循环异常")
    lifetime = ControlledWindowLifetime(window, FakeImeSession(events))
    lifetime.start(lambda: events.append("ready"))

    with pytest.raises(RuntimeError, match="主循环异常"):
        lifetime.run()

    assert events[-2:] == ["restore_ime", "destroy"]


def test_controlled_window_lifetime_surfaces_restore_failure_after_destroy() -> None:
    events: list[str] = []
    window = FakeWindow(events)
    ime_session = FakeImeSession(events)
    ime_session.restore_error = OSError("恢复 IME 失败")
    lifetime = ControlledWindowLifetime(window, ime_session)
    lifetime.start(lambda: events.append("ready"))

    with pytest.raises(OSError, match="恢复 IME 失败"):
        lifetime.run()

    assert events[-2:] == ["restore_ime", "destroy"]


def test_controlled_window_lifetime_surfaces_tk_callback_error_after_cleanup() -> None:
    events: list[str] = []
    window = FakeWindow(events)

    def fail_in_callback() -> None:
        events.append("callback")
        raise OSError("回调写盘失败")

    window.on_mainloop = fail_in_callback
    lifetime = ControlledWindowLifetime(window, FakeImeSession(events))
    lifetime.start(lambda: events.append("ready"))

    with pytest.raises(OSError, match="回调写盘失败"):
        lifetime.run()

    assert "tk_reported_callback_error" not in events
    assert events[-2:] == ["restore_ime", "destroy"]


def test_controlled_target_moves_with_wasd_and_clamps_to_canvas() -> None:
    model = ControlledTargetModel(
        width=100,
        height=80,
        start=(50, 40),
        goal=(90, 10),
        marker_radius=5,
        goal_radius=4,
        speed_px_per_second=100,
    )

    up = model.step(frozenset({"w"}), delta_seconds=0.1, elapsed_ms=0)
    corner = model.step(
        frozenset({"w", "d"}), delta_seconds=1, elapsed_ms=100
    )

    assert (up.x, up.y) == (50, 30)
    assert (corner.x, corner.y) == (95, 5)


def test_controlled_target_opposite_keys_cancel_each_other() -> None:
    model = ControlledTargetModel(
        width=100,
        height=80,
        start=(50, 40),
        goal=(90, 10),
        marker_radius=5,
        goal_radius=4,
        speed_px_per_second=100,
    )

    state = model.step(
        frozenset({"w", "s", "a", "d"}), delta_seconds=0.5, elapsed_ms=0
    )

    assert (state.x, state.y) == (50, 40)
    assert state.arrived is False


def test_controlled_target_can_ignore_initial_input_for_stuck_recovery_test() -> None:
    model = ControlledTargetModel(
        width=100,
        height=80,
        start=(50, 40),
        goal=(90, 10),
        marker_radius=5,
        goal_radius=4,
        speed_px_per_second=100,
        ignore_input_ms=300,
    )

    frozen = model.step(frozenset({"w"}), delta_seconds=0.1, elapsed_ms=299)
    moving = model.step(frozenset({"w"}), delta_seconds=0.1, elapsed_ms=300)

    assert (frozen.x, frozen.y) == (50, 40)
    assert (moving.x, moving.y) == (50, 30)


def test_controlled_target_arrival_is_sticky() -> None:
    model = ControlledTargetModel(
        width=100,
        height=80,
        start=(80, 10),
        goal=(90, 10),
        marker_radius=5,
        goal_radius=6,
        speed_px_per_second=100,
    )

    arrived = model.step(frozenset({"d"}), delta_seconds=0.05, elapsed_ms=0)
    moved_away = model.step(frozenset({"a"}), delta_seconds=0.5, elapsed_ms=50)

    assert arrived.arrived is True
    assert moved_away.arrived is True


def test_ground_truth_writer_appends_stable_jsonl(tmp_path) -> None:
    path = tmp_path / "ground-truth.jsonl"
    path.write_text(
        '{"event":"position","payload":{"arrived":true}}\n',
        encoding="utf-8",
    )
    writer = GroundTruthWriter(path, run_id="controlled-run")

    writer.write("start", at_ns=1, payload={"x": 10.5, "y": 20.5})
    writer.write("arrived", at_ns=2, payload={"success": True})

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == ["start", "arrived"]
    assert {record["run_id"] for record in records} == {"controlled-run"}
    assert records[0]["payload"] == {"x": 10.5, "y": 20.5}
