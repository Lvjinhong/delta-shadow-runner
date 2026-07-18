import pytest

from delta_vision.actuator import (
    DryRunActuator,
    DryRunInputStateError,
    ExpiredDryRunActionError,
)


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


def test_dry_run_mouse_move_records_delta_without_pressing_key() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)

    actuator.move_mouse_relative(12, -7, now_ns=1)

    event = actuator.events[0]
    assert (event.kind, event.key, event.dx, event.dy, event.dry_run) == (
        "mouse_move",
        None,
        12,
        -7,
        True,
    )
    assert actuator.pressed_keys == frozenset()


def test_dry_run_absolute_click_records_same_three_phase_audit_as_armed() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)

    actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert [event.kind for event in actuator.events] == [
        "mouse_move_absolute",
        "mouse_left_down",
        "mouse_left_up",
    ]
    assert [(event.x, event.y) for event in actuator.events] == [
        (500, 600),
        (None, None),
        (None, None),
    ]
    assert all(event.dry_run for event in actuator.events)


def test_dry_run_absolute_click_rejects_expired_intent_without_event() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)

    with pytest.raises(ExpiredDryRunActionError, match="过期"):
        actuator.click_left_at(500, 600, now_ns=200, expires_at_ns=200)

    assert actuator.events == ()


def test_dry_run_absolute_click_rejects_intent_older_than_audit_clock() -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)
    actuator.key_down("w", now_ns=300)
    events_before_click = actuator.events

    with pytest.raises(ExpiredDryRunActionError, match="过期"):
        actuator.click_left_at(500, 600, now_ns=100, expires_at_ns=200)

    assert actuator.events == events_before_click


def test_dry_run_key_tap_records_down_and_up_without_leaving_pressed_key() -> None:
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)

    actuator.tap_key("space", now_ns=100, expires_at_ns=200)

    assert [(event.kind, event.key) for event in actuator.events] == [
        ("key_down", "space"),
        ("key_up", "space"),
    ]
    assert actuator.events[-1].reason == "按键点击完成"
    assert actuator.pressed_keys == frozenset()


def test_dry_run_key_tap_rejects_stale_intent_against_audit_clock() -> None:
    actuator = DryRunActuator(allowed_keys={"w", "space"}, max_key_hold_ms=250)
    actuator.key_down("w", now_ns=300)
    events_before_tap = actuator.events

    with pytest.raises(ExpiredDryRunActionError, match="过期"):
        actuator.tap_key("space", now_ns=100, expires_at_ns=200)

    assert actuator.events == events_before_tap


def test_dry_run_key_tap_never_releases_preexisting_key_hold() -> None:
    actuator = DryRunActuator(allowed_keys={"space"}, max_key_hold_ms=250)
    actuator.key_down("space", now_ns=50)
    events_before_tap = actuator.events

    with pytest.raises(DryRunInputStateError, match="尚未释放"):
        actuator.tap_key("space", now_ns=100, expires_at_ns=200)

    assert actuator.events == events_before_tap
    assert actuator.pressed_keys == frozenset({"space"})


@pytest.mark.parametrize(
    ("screen_x", "screen_y", "now_ns", "expires_at_ns"),
    [
        (True, 600, 100, 200),
        (500, 1.5, 100, 200),
        (500, 600, True, 200),
        (500, 600, -1, 200),
        (500, 600, 100, True),
        (500, 600, 100, 0),
    ],
)
def test_dry_run_absolute_click_rejects_invalid_values_without_event(
    screen_x: object,
    screen_y: object,
    now_ns: object,
    expires_at_ns: object,
) -> None:
    actuator = DryRunActuator(allowed_keys={"w"}, max_key_hold_ms=250)

    with pytest.raises(ValueError):
        actuator.click_left_at(
            screen_x,  # type: ignore[arg-type]
            screen_y,  # type: ignore[arg-type]
            now_ns=now_ns,  # type: ignore[arg-type]
            expires_at_ns=expires_at_ns,  # type: ignore[arg-type]
        )

    assert actuator.events == ()


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


def test_dry_run_actuator_keeps_event_time_monotonic_after_late_release() -> None:
    actuator = DryRunActuator(allowed_keys={"w", "d"}, max_key_hold_ms=250)
    actuator.key_down("w", now_ns=0)
    actuator.key_up("w", now_ns=300_000_000)

    actuator.key_down("d", now_ns=50_000_000)

    assert [event.at_ns for event in actuator.events] == [0, 300_000_000, 300_000_000]


@pytest.mark.parametrize(
    ("allowed_keys", "max_key_hold_ms", "message"),
    [(set(), 250, "允许按键集合"), ({"w"}, 0, "最大按键时长")],
)
def test_dry_run_actuator_rejects_invalid_configuration(
    allowed_keys: set[str], max_key_hold_ms: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DryRunActuator(allowed_keys=allowed_keys, max_key_hold_ms=max_key_hold_ms)
