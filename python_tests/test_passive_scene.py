from collections.abc import Iterable

import numpy as np
import pytest

from delta_vision.config import CaptureRegion
from delta_vision.frames import CapturedFrame
from delta_vision.menu_automation import (
    MenuScene,
    SceneDecisionReason,
    SceneObservation,
)
from delta_vision.passive_scene import (
    PassiveReturnPolicy,
    PassiveReturnStatus,
    _ForegroundGuardedSource,
    run_passive_return_loop,
)


class _Source:
    def __init__(self, frames: Iterable[CapturedFrame | None]) -> None:
        self._frames = iter(frames)
        self.closed = False

    def grab(self) -> CapturedFrame | None:
        return next(self._frames, None)

    def close(self) -> None:
        self.closed = True


class _FailingSource:
    def grab(self) -> CapturedFrame | None:
        raise RuntimeError("grab boom")

    def close(self) -> None:
        raise RuntimeError("close boom")


class _Observer:
    def observe(self, frame: CapturedFrame) -> SceneObservation:
        raw_scene = frame.metadata.get("scene")
        reason = frame.metadata.get("reason", SceneDecisionReason.ACCEPTED)
        if raw_scene is None:
            return SceneObservation(
                frame_sequence=frame.sequence,
                captured_at_ns=frame.captured_at_ns,
                scene=MenuScene.UNKNOWN,
                candidate_scene=None,
                confidence=0.0,
                runner_up_confidence=0.0,
                accepted=False,
                reason=reason,
                action_accepted=False,
                action_point=None,
                page_point=None,
                page_template_id=None,
            )
        scene = MenuScene(raw_scene)
        return SceneObservation(
            frame_sequence=frame.sequence,
            captured_at_ns=frame.captured_at_ns,
            scene=scene,
            candidate_scene=scene,
            confidence=0.99,
            runner_up_confidence=0.1,
            accepted=True,
            reason=SceneDecisionReason.ACCEPTED,
            action_accepted=False,
            action_point=None,
            page_point=(10.0, 10.0),
            page_template_id=f"{scene.value}-page",
        )


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        self.now_ns += 10_000_000
        return self.now_ns


class _Gateway:
    def __init__(self, handles: Iterable[int], titles: Iterable[str]) -> None:
        self._handles = iter(handles)
        self._titles = iter(titles)
        self._last_handle = 0
        self._last_title = ""

    def foreground_window_handle(self) -> int:
        self._last_handle = next(self._handles, self._last_handle)
        return self._last_handle

    def foreground_title(self) -> str:
        self._last_title = next(self._titles, self._last_title)
        return self._last_title


def _frame(
    sequence: int,
    scene: MenuScene | None,
    *,
    captured_at_ns: int | None = None,
    reason: SceneDecisionReason = SceneDecisionReason.ACCEPTED,
) -> CapturedFrame:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image.setflags(write=False)
    metadata = {"scene": None if scene is None else scene.value, "reason": reason}
    return CapturedFrame(
        sequence=sequence,
        captured_at_ns=(
            900_000_000 + sequence * 10_000_000 if captured_at_ns is None else captured_at_ns
        ),
        image=image,
        source="passive-test",
        metadata=metadata,
    )


def _policy(**overrides: object) -> PassiveReturnPolicy:
    values: dict[str, object] = {
        "target_scene": MenuScene.BASE,
        "allowed_transient_scenes": frozenset(
            {MenuScene.IN_MATCH, MenuScene.DEATH_SUMMARY, MenuScene.POST_MATCH}
        ),
        "confirmation_frames": 3,
        "maximum_confirmation_span_ms": 500,
        "maximum_frame_age_ms": 500,
        "loop_interval_ms": 1,
        "max_duration_seconds": 1,
    }
    values.update(overrides)
    return PassiveReturnPolicy(**values)


def test_passive_return_accepts_post_match_then_three_base_frames() -> None:
    source = _Source(
        [
            _frame(1, MenuScene.POST_MATCH),
            _frame(2, MenuScene.BASE),
            _frame(3, MenuScene.BASE),
            _frame(4, MenuScene.BASE),
        ]
    )

    result = run_passive_return_loop(
        source=source,
        observer=_Observer(),
        policy=_policy(),
        clock_ns=_Clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is PassiveReturnStatus.BASE_CONFIRMED
    assert result.terminal_scene is MenuScene.BASE
    assert result.seen_scenes == (
        MenuScene.POST_MATCH,
        MenuScene.BASE,
        MenuScene.BASE,
        MenuScene.BASE,
    )
    assert result.frame_count == 4
    assert source.closed is True


def test_unknown_breaks_base_confirmation_and_never_advances() -> None:
    source = _Source(
        [
            _frame(1, MenuScene.BASE),
            _frame(2, MenuScene.BASE),
            _frame(3, None, reason=SceneDecisionReason.AMBIGUOUS),
            _frame(4, MenuScene.BASE),
            _frame(5, MenuScene.BASE),
        ]
    )

    result = run_passive_return_loop(
        source=source,
        observer=_Observer(),
        policy=_policy(max_duration_seconds=0.08),
        clock_ns=_Clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is PassiveReturnStatus.STOPPED
    assert result.terminal_scene is MenuScene.UNKNOWN
    assert "超时" in (result.reason or "")
    assert source.closed is True


def test_unexpected_confirmed_scene_stops_closed() -> None:
    source = _Source([_frame(1, MenuScene.STRATEGY_BOARD)])

    result = run_passive_return_loop(
        source=source,
        observer=_Observer(),
        policy=_policy(),
        clock_ns=_Clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is PassiveReturnStatus.STOPPED
    assert result.terminal_scene is MenuScene.STRATEGY_BOARD
    assert "未允许" in (result.reason or "")


def test_frame_size_mismatch_stops_immediately() -> None:
    source = _Source(
        [
            _frame(
                1,
                None,
                reason=SceneDecisionReason.FRAME_SIZE_MISMATCH,
            )
        ]
    )

    result = run_passive_return_loop(
        source=source,
        observer=_Observer(),
        policy=_policy(),
        clock_ns=_Clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is PassiveReturnStatus.STOPPED
    assert "分辨率" in (result.reason or "")


def test_duplicate_sequence_stops_closed() -> None:
    source = _Source([_frame(1, MenuScene.POST_MATCH), _frame(1, MenuScene.BASE)])

    result = run_passive_return_loop(
        source=source,
        observer=_Observer(),
        policy=_policy(),
        clock_ns=_Clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is PassiveReturnStatus.STOPPED
    assert "严格递增" in (result.reason or "")


def test_stale_frame_stops_closed() -> None:
    source = _Source([_frame(1, MenuScene.POST_MATCH, captured_at_ns=1)])

    result = run_passive_return_loop(
        source=source,
        observer=_Observer(),
        policy=_policy(maximum_frame_age_ms=50),
        clock_ns=_Clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert result.status is PassiveReturnStatus.STOPPED
    assert "过期" in (result.reason or "")


def test_invalid_initial_clock_still_closes_source() -> None:
    source = _Source([])

    with pytest.raises(ValueError, match="时钟"):
        run_passive_return_loop(
            source=source,
            observer=_Observer(),
            policy=_policy(),
            clock_ns=lambda: -1,
            sleep_fn=lambda _seconds: None,
        )

    assert source.closed is True


def test_guard_closes_when_client_region_changes_during_capture() -> None:
    expected_region = CaptureRegion(100, 200, 96, 64)
    moved_region = CaptureRegion(101, 200, 96, 64)
    regions = iter((expected_region, moved_region))
    source = _Source([_frame(1, MenuScene.POST_MATCH)])
    guarded = _ForegroundGuardedSource(
        source,
        gateway=_Gateway((7, 7), ("三角洲行动  ", "三角洲行动  ")),
        target_window_handle=7,
        target_window_title="三角洲行动  ",
        expected_region=expected_region,
        region_resolver=lambda _title: next(regions),
    )

    with pytest.raises(RuntimeError, match="客户区"):
        run_passive_return_loop(
            source=guarded,
            observer=_Observer(),
            policy=_policy(),
            clock_ns=_Clock(),
            sleep_fn=lambda _seconds: None,
        )

    assert source.closed is True


def test_guard_closes_when_window_loses_focus_during_capture() -> None:
    expected_region = CaptureRegion(100, 200, 96, 64)
    source = _Source([_frame(1, MenuScene.POST_MATCH)])
    guarded = _ForegroundGuardedSource(
        source,
        gateway=_Gateway((7, 8), ("三角洲行动  ",)),
        target_window_handle=7,
        target_window_title="三角洲行动  ",
        expected_region=expected_region,
        region_resolver=lambda _title: expected_region,
    )

    with pytest.raises(RuntimeError, match="前台窗口"):
        run_passive_return_loop(
            source=guarded,
            observer=_Observer(),
            policy=_policy(),
            clock_ns=_Clock(),
            sleep_fn=lambda _seconds: None,
        )

    assert source.closed is True


def test_close_failure_does_not_hide_primary_observation_error() -> None:
    with pytest.raises(RuntimeError, match="grab boom") as raised:
        run_passive_return_loop(
            source=_FailingSource(),
            observer=_Observer(),
            policy=_policy(),
            clock_ns=_Clock(),
            sleep_fn=lambda _seconds: None,
        )

    assert any("close boom" in note for note in (raised.value.__notes__ or ()))
