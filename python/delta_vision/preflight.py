"""汇总 Windows 截图与受控 E2E 产物，生成机器可判定的前置验收报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .benchmark import CaptureMetrics, evaluate_capture_metrics
from .worker import load_worker_settings


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    id: str
    passed: bool
    detail: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_mapping(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return raw


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL 第 {line_number} 行必须是对象: {path}")
        records.append(record)
    if not records:
        raise ValueError(f"JSONL 不能为空: {path}")
    return records


def _payload(record: Mapping[str, object]) -> Mapping[str, object]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _validate_event_records(
    records: Sequence[Mapping[str, object]],
    *,
    expected_run_id: str,
    event_field: str,
) -> str | None:
    previous_at_ns = -1
    for index, record in enumerate(records, start=1):
        if type(record.get("schema_version")) is not int or record.get(
            "schema_version"
        ) != 1:
            return f"第 {index} 条事件 schema_version 非法"
        if record.get("run_id") != expected_run_id:
            return f"第 {index} 条事件 run_id 不匹配"
        at_ns = record.get("at_ns")
        if type(at_ns) is not int or at_ns < 0:
            return f"第 {index} 条事件 at_ns 非法"
        if at_ns < previous_at_ns:
            return f"第 {index} 条事件 at_ns 倒退"
        previous_at_ns = at_ns
        event_name = record.get(event_field)
        if not isinstance(event_name, str) or not event_name:
            return f"第 {index} 条事件 {event_field} 非法"
        if not isinstance(record.get("payload"), dict):
            return f"第 {index} 条事件 payload 非法"
    return None


def _artifact_freshness(
    paths: Sequence[Path],
    *,
    now_seconds: float,
    max_age_seconds: float,
) -> tuple[bool, str]:
    failures: list[str] = []
    for path in paths:
        try:
            age_seconds = now_seconds - path.stat().st_mtime
        except OSError as error:
            failures.append(f"{path.name}: {error}")
            continue
        if age_seconds < -5:
            failures.append(f"{path.name}: mtime 位于未来")
        elif age_seconds > max_age_seconds:
            failures.append(f"{path.name}: 已过期 {age_seconds:.1f}s")
    return (not failures, "fresh" if not failures else "; ".join(failures))


def _evaluate_worker_events(
    records: Sequence[Mapping[str, object]],
) -> tuple[bool, bool, bool, str]:
    held_keys: set[str] = set()
    saw_key_down = False
    saw_key_up = False
    state_error: str | None = None
    for index, record in enumerate(records, start=1):
        if record.get("event_type") == "input":
            payload = _payload(record)
            kind = payload.get("kind")
            if kind not in {"key_down", "key_up", "mouse_move"}:
                state_error = f"第 {index} 条 input kind 非法"
                break
            if kind == "mouse_move":
                continue
            key = payload.get("key")
            if not isinstance(key, str) or not key:
                state_error = f"第 {index} 条 input key 非法"
                break
            if kind == "key_down":
                if key in held_keys:
                    state_error = f"第 {index} 条重复 key_down"
                    break
                held_keys.add(key)
                saw_key_down = True
            else:
                if key not in held_keys:
                    state_error = f"第 {index} 条 key_up 没有对应 key_down"
                    break
                held_keys.remove(key)
                saw_key_up = True
        elif record.get("event_type") == "frame":
            pressed_keys = _payload(record).get("pressed_keys")
            if pressed_keys != sorted(held_keys):
                state_error = f"第 {index} 条 frame 的 pressed_keys 与输入重放不一致"
                break

    terminal_record = records[-1]
    terminal_payload = _payload(terminal_record)
    terminal_order = terminal_record.get("event_type") == "frame"
    worker_arrived = terminal_order and (
        terminal_payload.get("status") == "arrived"
        and terminal_payload.get("pressed_keys") == []
    )
    input_valid = (
        state_error is None and saw_key_down and saw_key_up and not held_keys
    )
    detail = state_error or (
        "input replay released" if input_valid else "input replay incomplete"
    )
    return worker_arrived, input_valid, terminal_order, detail


def evaluate_preflight_artifacts(
    *,
    run_id: str,
    config_path: Path,
    capture_metrics_path: Path,
    capture_gate_path: Path,
    worker_events_path: Path,
    ground_truth_path: Path,
    process_exit_codes: tuple[int, int, int],
    max_artifact_age_seconds: float = 900,
    now_seconds: float | None = None,
) -> dict[str, object]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id 必须是非空字符串")
    if max_artifact_age_seconds <= 0:
        raise ValueError("max_artifact_age_seconds 必须大于 0")
    generated_at_seconds = time.time() if now_seconds is None else now_seconds

    config_sha256: str | None = None
    config_error: str | None = None
    try:
        config_sha256 = _sha256(config_path)
        settings = load_worker_settings(config_path)
    except Exception as error:
        settings = None
        config_error = str(error)
    checks: list[PreflightCheck] = [
        PreflightCheck(
            id="config",
            passed=settings is not None,
            detail=(
                config_error
                or (
                    f"{settings.target_window_title} / {settings.capture_backend} / "
                    f"goal={settings.goal_node_id}"
                    if settings is not None
                    else "invalid"
                )
            ),
        )
    ]

    exit_codes = {
        "config": process_exit_codes[0],
        "controlled": process_exit_codes[1],
        "benchmark": process_exit_codes[2],
    }
    checks.append(
        PreflightCheck(
            id="process_exit_codes",
            passed=all(type(code) is int and code == 0 for code in exit_codes.values()),
            detail=json.dumps(exit_codes, ensure_ascii=False, sort_keys=True),
        )
    )

    first_frame_path = capture_metrics_path.parent / "first-frame.png"
    last_frame_path = capture_metrics_path.parent / "last-frame.png"
    evidence_paths = (
        capture_metrics_path,
        capture_gate_path,
        worker_events_path,
        ground_truth_path,
        first_frame_path,
        last_frame_path,
    )
    freshness_passed, freshness_detail = _artifact_freshness(
        evidence_paths,
        now_seconds=generated_at_seconds,
        max_age_seconds=max_artifact_age_seconds,
    )
    checks.append(
        PreflightCheck(
            id="artifact_freshness",
            passed=freshness_passed,
            detail=freshness_detail,
        )
    )

    metrics_raw: dict[str, object] | None = None
    metrics: CaptureMetrics | None = None
    computed_gate = None
    capture_error: str | None = None
    try:
        metrics_raw = _read_mapping(capture_metrics_path)
        if isinstance(metrics_raw.get("initial_resolution"), list):
            metrics_raw["initial_resolution"] = tuple(metrics_raw["initial_resolution"])
        metrics = CaptureMetrics(**metrics_raw)
        computed_gate = evaluate_capture_metrics(metrics)
    except Exception as error:
        capture_error = str(error)

    gate_error: str | None = None
    try:
        persisted_gate = _read_mapping(capture_gate_path)
    except Exception as error:
        persisted_gate = None
        gate_error = str(error)
    try:
        worker_events = _read_jsonl(worker_events_path)
        worker_error = _validate_event_records(
            worker_events,
            expected_run_id=run_id,
            event_field="event_type",
        )
    except Exception as error:
        worker_events = []
        worker_error = str(error)
    try:
        ground_truth_records = _read_jsonl(ground_truth_path)
        ground_truth_error = _validate_event_records(
            ground_truth_records,
            expected_run_id=run_id,
            event_field="event",
        )
    except Exception as error:
        ground_truth_records = []
        ground_truth_error = str(error)

    run_binding = (
        metrics is not None
        and metrics.run_id == run_id
        and persisted_gate is not None
        and persisted_gate.get("run_id") == run_id
        and worker_error is None
        and ground_truth_error is None
    )
    checks.append(
        PreflightCheck(
            id="run_binding",
            passed=run_binding,
            detail="matched" if run_binding else "run_id 或事件 schema 不匹配",
        )
    )

    checks.append(
        PreflightCheck(
            id="capture",
            passed=computed_gate is not None and computed_gate.passed,
            detail=(
                capture_error
                or (
                    "passed"
                    if computed_gate is not None and computed_gate.passed
                    else ",".join(computed_gate.failures if computed_gate else ())
                )
            ),
        )
    )
    expected_gate = (
        {
            "schema_version": computed_gate.schema_version,
            "run_id": computed_gate.run_id,
            "passed": computed_gate.passed,
            "failures": list(computed_gate.failures),
        }
        if computed_gate is not None
        else None
    )
    gate_consistent = persisted_gate is not None and persisted_gate == expected_gate
    checks.append(
        PreflightCheck(
            id="capture_gate_consistency",
            passed=gate_consistent,
            detail=(
                "matched"
                if gate_consistent
                else gate_error or capture_error or "persisted gate differs"
            ),
        )
    )

    frame_hashes_passed = False
    frame_hash_detail = capture_error or "capture metrics unavailable"
    if metrics is not None:
        try:
            frame_hashes_passed = (
                metrics.first_frame_sha256 == _sha256(first_frame_path)
                and metrics.last_frame_sha256 == _sha256(last_frame_path)
            )
            frame_hash_detail = "matched" if frame_hashes_passed else "frame hash differs"
        except OSError as error:
            frame_hash_detail = str(error)
    checks.append(
        PreflightCheck(
            id="capture_frame_hashes",
            passed=frame_hashes_passed,
            detail=frame_hash_detail,
        )
    )

    if worker_error is None:
        worker_arrived, input_valid, terminal_order, input_detail = (
            _evaluate_worker_events(worker_events)
        )
    else:
        worker_arrived = input_valid = terminal_order = False
        input_detail = worker_error
    checks.extend(
        (
            PreflightCheck(
                id="controlled_worker",
                passed=worker_arrived,
                detail=("arrived and released" if worker_arrived else input_detail),
            ),
            PreflightCheck(
                id="controlled_input",
                passed=input_valid,
                detail=input_detail,
            ),
            PreflightCheck(
                id="controlled_terminal_order",
                passed=terminal_order,
                detail=("terminal frame is final" if terminal_order else input_detail),
            ),
        )
    )

    ground_truth_arrived = False
    arrival_at_ns: int | None = None
    if ground_truth_error is None:
        start_indices = [
            index
            for index, record in enumerate(ground_truth_records)
            if record.get("event") == "start"
        ]
        if start_indices == [0]:
            for record in ground_truth_records[1:]:
                if (
                    record.get("event") == "position"
                    and _payload(record).get("arrived") is True
                ):
                    ground_truth_arrived = True
                    arrival_at_ns = record["at_ns"]  # 已由 schema 校验为 int
                    break
    checks.append(
        PreflightCheck(
            id="controlled_ground_truth",
            passed=ground_truth_arrived,
            detail=(
                "position arrived after sole start"
                if ground_truth_arrived
                else ground_truth_error or "合法 position arrival 缺失"
            ),
        )
    )

    timing_passed = False
    if worker_error is None and ground_truth_error is None and arrival_at_ns is not None:
        worker_start = worker_events[0]["at_ns"]
        worker_end = worker_events[-1]["at_ns"]
        truth_start = ground_truth_records[0]["at_ns"]
        truth_end = ground_truth_records[-1]["at_ns"]
        timing_passed = (
            worker_start <= truth_end
            and truth_start <= worker_end
            and arrival_at_ns <= worker_end
        )
    checks.append(
        PreflightCheck(
            id="controlled_timing",
            passed=timing_passed,
            detail="ranges overlap" if timing_passed else "单调时钟范围不相交",
        )
    )

    evidence: dict[str, dict[str, object]] = {}
    for path in evidence_paths:
        try:
            evidence[path.name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "mtime": path.stat().st_mtime,
            }
        except OSError:
            continue

    return {
        "schema_version": 2,
        "run_id": run_id,
        "generated_at_utc": datetime.fromtimestamp(
            generated_at_seconds, tz=UTC
        ).isoformat(),
        "passed": all(check.passed for check in checks),
        "config": str(config_path),
        "config_sha256": config_sha256,
        "process_exit_codes": exit_codes,
        "evidence": evidence,
        "checks": [asdict(check) for check in checks],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows 外部视觉运行前置证据校验")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--capture-metrics", type=Path, required=True)
    parser.add_argument("--capture-gate", type=Path, required=True)
    parser.add_argument("--worker-events", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--config-exit-code", type=int, required=True)
    parser.add_argument("--controlled-exit-code", type=int, required=True)
    parser.add_argument("--benchmark-exit-code", type=int, required=True)
    parser.add_argument("--max-artifact-age-seconds", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate_preflight_artifacts(
            run_id=args.run_id,
            config_path=args.config,
            capture_metrics_path=args.capture_metrics,
            capture_gate_path=args.capture_gate,
            worker_events_path=args.worker_events,
            ground_truth_path=args.ground_truth,
            process_exit_codes=(
                args.config_exit_code,
                args.controlled_exit_code,
                args.benchmark_exit_code,
            ),
            max_artifact_age_seconds=args.max_artifact_age_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as error:
        print(f"前置验收证据校验失败: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
