import pytest

from delta_vision.actuator import DryRunActuator


def test_dry_run_actuator_tracks_pressed_keys_and_releases_all() -> None:
    actuator = DryRunActuator(allowed_keys={"w", "a", "d"}, max_key_hold_ms=250)

    actuator.key_down("w", now_ns=1_000_000_000)
    actuator.key_down("a", now_ns=1_010_000_000)
    assert actuator.pressed_keys == frozenset({"w", "a"})

    actuator.release_all(now_ns=1_020_000_000, reason="测试结束")

    assert actuator.pressed_keys == frozenset()
    assert [event.kind for event in actuator.events] == ["key_down", "key_down", "key_up", "key_up"]
    assert [event.key for event in actuator.events[-2:]] == ["a", "w"]
    assert all(event.dry_run for event in actuator.events)


def test_dry_run_actuator_rejects_unknown_key() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)

    with pytest.raises(ValueError, match="不允许的按键"):
        actuator.key_down("x", now_ns=1)


def test_dry_run_actuator_expires_overdue_key_holds() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)
    actuator.key_down("w", now_ns=1_000_000_000)

    expired = actuator.expire_overdue(now_ns=1_251_000_000)

    assert expired == ("w",)
    assert actuator.pressed_keys == frozenset()
    assert actuator.events[-1].reason == "超过最大按键时长"


def test_dry_run_actuator_key_down_is_idempotent() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)

    actuator.key_down("w", now_ns=1)
    actuator.key_down("w", now_ns=2)

    assert actuator.pressed_keys == frozenset({"w"})
    assert len(actuator.events) == 1


def test_dry_run_actuator_key_up_and_repeated_release_are_idempotent() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)
    actuator.key_down("w", now_ns=1)

    actuator.key_up("w", now_ns=2, reason="正常释放")
    actuator.key_up("w", now_ns=3, reason="重复释放")
    actuator.release_all(now_ns=4, reason="清理")

    assert actuator.pressed_keys == frozenset()
    assert [event.kind for event in actuator.events] == ["key_down", "key_up"]
    assert actuator.events[-1].reason == "正常释放"


@pytest.mark.parametrize(
    ("allowed_keys", "max_key_hold_ms", "message"),
    [(set(), 250, "允许按键集合"), ({"w"}, 0, "最大按键时长")],
)
def test_dry_run_actuator_rejects_invalid_configuration(
    allowed_keys: set[str], max_key_hold_ms: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DryRunActuator(allowed_keys=allowed_keys, max_key_hold_ms=max_key_hold_ms)
