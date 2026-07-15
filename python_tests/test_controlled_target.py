import json

from delta_vision.controlled_target import (
    ControlledTargetModel,
    GroundTruthWriter,
)


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
    writer = GroundTruthWriter(tmp_path / "ground-truth.jsonl")

    writer.write("start", at_ns=1, payload={"x": 10.5, "y": 20.5})
    writer.write("arrived", at_ns=2, payload={"success": True})

    records = [
        json.loads(line)
        for line in (tmp_path / "ground-truth.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["event"] for record in records] == ["start", "arrived"]
    assert records[0]["payload"] == {"x": 10.5, "y": 20.5}
