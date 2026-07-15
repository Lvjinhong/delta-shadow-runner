import hashlib
import json
import os
from pathlib import Path

import pytest

from delta_vision.preflight import main as preflight_main

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "controlled-window.json"
RUN_ID = "preflight-test-run"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def _write_preflight_evidence(
    root: Path, *, black_frame_count: int = 0
) -> dict[str, Path]:
    paths = {
        "metrics": root / "capture-metrics.json",
        "gate": root / "capture-gate.json",
        "worker": root / "events.jsonl",
        "ground_truth": root / "target-ground-truth.jsonl",
        "first_frame": root / "first-frame.png",
        "last_frame": root / "last-frame.png",
        "report": root / "preflight-report.json",
    }
    paths["first_frame"].write_bytes(b"first-frame")
    paths["last_frame"].write_bytes(b"last-frame")
    _write_json(
        paths["metrics"],
        {
            "schema_version": 2,
            "run_id": RUN_ID,
            "duration_seconds": 60,
            "frame_count": 1200,
            "no_frame_count": 0,
            "black_frame_count": black_frame_count,
            "average_fps": 20,
            "capture_latency_average_ms": 20,
            "capture_latency_p95_ms": 40,
            "capture_latency_max_ms": 45,
            "initial_resolution": [1920, 1080],
            "resolution_drift_count": 0,
            "foreground_mismatch_count": 0,
            "started_at_ns": 1_000_000_000,
            "ended_at_ns": 61_000_000_000,
            "first_frame_sha256": hashlib.sha256(b"first-frame").hexdigest(),
            "last_frame_sha256": hashlib.sha256(b"last-frame").hexdigest(),
        },
    )
    _write_json(
        paths["gate"],
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "passed": black_frame_count == 0,
            "failures": [] if black_frame_count == 0 else ["black_frames"],
        },
    )
    _write_jsonl(
        paths["worker"],
        [
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "event_type": "input",
                "at_ns": 120,
                "payload": {"kind": "key_down", "key": "w"},
            },
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "event_type": "input",
                "at_ns": 130,
                "payload": {"kind": "key_up", "key": "w"},
            },
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "event_type": "frame",
                "at_ns": 140,
                "payload": {"status": "arrived", "pressed_keys": []},
            },
        ],
    )
    _write_jsonl(
        paths["ground_truth"],
        [
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "event": "start",
                "at_ns": 100,
                "payload": {"arrived": False},
            },
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "event": "position",
                "at_ns": 135,
                "payload": {"arrived": True},
            },
        ],
    )
    return paths


def _preflight_arguments(
    paths: dict[str, Path],
    *,
    config_path: Path = CONFIG_PATH,
    process_exit_codes: tuple[int, int, int] = (0, 0, 0),
) -> list[str]:
    return [
        "--run-id",
        RUN_ID,
        "--config",
        str(config_path),
        "--capture-metrics",
        str(paths["metrics"]),
        "--capture-gate",
        str(paths["gate"]),
        "--worker-events",
        str(paths["worker"]),
        "--ground-truth",
        str(paths["ground_truth"]),
        "--config-exit-code",
        str(process_exit_codes[0]),
        "--controlled-exit-code",
        str(process_exit_codes[1]),
        "--benchmark-exit-code",
        str(process_exit_codes[2]),
        "--output",
        str(paths["report"]),
    ]


def test_preflight_cli_accepts_complete_capture_and_controlled_evidence(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema_version"] == 2
    assert report["passed"] is True
    assert report["config_sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert {check["id"]: check["passed"] for check in report["checks"]} == {
        "config": True,
        "process_exit_codes": True,
        "artifact_freshness": True,
        "run_binding": True,
        "capture": True,
        "capture_gate_consistency": True,
        "capture_frame_hashes": True,
        "controlled_worker": True,
        "controlled_input": True,
        "controlled_terminal_order": True,
        "controlled_ground_truth": True,
        "controlled_timing": True,
    }


def test_preflight_cli_preserves_report_when_controlled_artifact_is_missing(
    tmp_path,
) -> None:
    paths = _write_preflight_evidence(tmp_path)
    paths["worker"].unlink()
    paths["ground_truth"].unlink()

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert report["passed"] is False
    assert checks["controlled_worker"] is False
    assert checks["controlled_input"] is False
    assert checks["controlled_ground_truth"] is False
    assert checks["capture"] is True


def test_preflight_cli_preserves_report_when_capture_artifacts_are_missing(
    tmp_path,
) -> None:
    paths = _write_preflight_evidence(tmp_path)
    paths["metrics"].unlink()
    paths["gate"].unlink()

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert report["passed"] is False
    assert checks["capture"] is False
    assert checks["capture_gate_consistency"] is False
    assert checks["controlled_worker"] is True


def test_preflight_cli_preserves_report_when_config_is_invalid(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)
    config_path = tmp_path / "invalid-config.json"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["capture_backend"] = "invalid"
    _write_json(config_path, config)

    exit_code = preflight_main(
        _preflight_arguments(paths, config_path=config_path)
    )

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert report["passed"] is False
    assert checks["config"] is False
    assert checks["capture"] is True
    assert checks["controlled_worker"] is True


def test_preflight_cli_rejects_nonzero_subprocess_exit_code(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)

    exit_code = preflight_main(
        _preflight_arguments(paths, process_exit_codes=(0, 4, 0))
    )

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert report["passed"] is False
    assert report["process_exit_codes"]["controlled"] == 4
    assert checks["process_exit_codes"] is False


def test_preflight_cli_rejects_stale_artifact_from_old_run(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)
    os.utime(paths["ground_truth"], (1, 1))

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert report["passed"] is False
    assert checks["artifact_freshness"] is False


def test_preflight_cli_rejects_input_after_terminal_frame(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)
    with paths["worker"].open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": RUN_ID,
                    "event_type": "input",
                    "at_ns": 150,
                    "payload": {"kind": "key_down", "key": "w"},
                }
            )
            + "\n"
        )

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert checks["controlled_terminal_order"] is False


def test_preflight_cli_rejects_unrelated_arrived_ground_truth_event(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)
    _write_jsonl(
        paths["ground_truth"],
        [
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "event": "start",
                "at_ns": 100,
                "payload": {},
            },
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "event": "unrelated",
                "at_ns": 135,
                "payload": {"arrived": True},
            },
        ],
    )

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert checks["controlled_ground_truth"] is False


def test_preflight_cli_rejects_mixed_run_ids(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    metrics["run_id"] = "old-run"
    _write_json(paths["metrics"], metrics)

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert checks["run_binding"] is False


def test_preflight_cli_rejects_capture_frame_hash_mismatch(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)
    paths["last_frame"].write_bytes(b"tampered")

    exit_code = preflight_main(_preflight_arguments(paths))

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checks = {check["id"]: check["passed"] for check in report["checks"]}
    assert exit_code == 2
    assert checks["capture_frame_hashes"] is False


def test_preflight_cli_requires_subprocess_exit_codes(tmp_path) -> None:
    paths = _write_preflight_evidence(tmp_path)
    arguments = _preflight_arguments(paths)
    for flag in (
        "--config-exit-code",
        "--controlled-exit-code",
        "--benchmark-exit-code",
    ):
        index = arguments.index(flag)
        del arguments[index : index + 2]

    with pytest.raises(SystemExit):
        preflight_main(arguments)
