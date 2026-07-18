import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from delta_vision.frames import CapturedFrame, DatasetContentDigest
from delta_vision.menu_automation import MenuControllerStatus, MenuScene
from delta_vision.menu_profile import load_menu_profile


def _write_template(path: Path, seed: int) -> tuple[np.ndarray, str]:
    image = np.random.default_rng(seed).integers(
        0,
        256,
        size=(8, 10, 3),
        dtype=np.uint8,
    )
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    raw = payload.tobytes()
    path.write_bytes(raw)
    return image, hashlib.sha256(raw).hexdigest()


def _detector(image_path: str, sha256: str) -> dict[str, object]:
    return {
        "image": image_path,
        "sha256": sha256,
        "search_roi": {"left": 0, "top": 0, "width": 96, "height": 64},
        "scales": [1.0],
        "score_threshold": 0.9,
        "minimum_margin": 0.08,
        "nms_radius_px": 8,
    }


def _write_profile(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    lobby_page, lobby_page_hash = _write_template(tmp_path / "lobby-page.png", 1)
    lobby_action, lobby_action_hash = _write_template(tmp_path / "lobby-action.png", 2)
    post_page, post_page_hash = _write_template(tmp_path / "post-page.png", 3)
    source_root = tmp_path / "source" / "cal-01"
    frames_root = source_root / "frames"
    frames_root.mkdir(parents=True)
    source_frames = [
        _frame(lobby_page, lobby_action).image,
        np.pad(
            post_page,
            ((10, 64 - 10 - post_page.shape[0]), (25, 96 - 25 - post_page.shape[1]), (0, 0)),
        ),
    ]
    manifest_records = []
    content_digest = DatasetContentDigest()
    frame_hashes: list[str] = []
    for sequence, image in enumerate(source_frames):
        frame_path = frames_root / f"frame-{sequence:08d}.png"
        assert cv2.imwrite(str(frame_path), image)
        frame_hashes.append(content_digest.update(sequence, image))
        manifest_records.append(
            {
                "captured_at_ns": (sequence + 1) * 100_000_000,
                "height": 64,
                "image": f"frames/{frame_path.name}",
                "metadata": {
                    "dataset_split": "calibration",
                    "run_id": "cal-01",
                },
                "sequence": sequence,
                "source": "profile-test",
                "width": 96,
            }
        )
    run_path = source_root / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "run_id": "cal-01",
                "dataset_split": "calibration",
                "frame_count": 2,
                "resolution": [96, 64],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = source_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in manifest_records),
        encoding="utf-8",
    )
    profile = {
        "schema_version": 1,
        "profile_id": "menu-test-v1",
        "expected_frame_size": [96, 64],
        "minimum_scene_margin": 0.08,
        "controller": {
            "confirmation_frames": 3,
            "maximum_confirmation_span_ms": 750,
            "maximum_action_point_drift_px": 12,
            "maximum_page_point_drift_px": 12,
            "maximum_frame_age_ms": 250,
            "transition_timeout_ms": 8000,
        },
        "source_runs": [
            {
                "run_id": "cal-01",
                "split": "calibration",
                "directory": "source/cal-01",
                "run_json_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                "frame_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "dataset_content_sha256": content_digest.hexdigest(),
            }
        ],
        "templates": [
            {
                "template_id": "lobby-preparation",
                "scene": "lobby",
                "source_run_id": "cal-01",
                "source_sequence": 0,
                "source_frame_sha256": frame_hashes[0],
                "page": _detector("lobby-page.png", lobby_page_hash),
                "action": _detector("lobby-action.png", lobby_action_hash),
                "action_region": {
                    "left": 40,
                    "top": 20,
                    "width": 50,
                    "height": 40,
                },
            },
            {
                "template_id": "post-match-progress",
                "scene": "post_match",
                "source_run_id": "cal-01",
                "source_sequence": 1,
                "source_frame_sha256": frame_hashes[1],
                "page": _detector("post-page.png", post_page_hash),
                "action": None,
                "action_region": None,
            },
        ],
        "transitions": [
            {
                "source": "lobby",
                "target": "post_match",
                "action_kind": "click",
                "key": None,
            }
        ],
        "stop_scenes": ["post_match"],
    }
    path = tmp_path / "menu-profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path, lobby_page, lobby_action


def _frame(page: np.ndarray, action: np.ndarray) -> CapturedFrame:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[5:13, 10:20] = page
    image[30:38, 50:60] = action
    image.setflags(write=False)
    return CapturedFrame(
        sequence=1,
        captured_at_ns=100_000_000,
        image=image,
        source="profile-test",
    )


def test_load_menu_profile_builds_real_observer_and_fresh_controller(
    tmp_path: Path,
) -> None:
    path, lobby_page, lobby_action = _write_profile(tmp_path)

    loaded = load_menu_profile(path)
    observation = loaded.observer.observe(_frame(lobby_page, lobby_action))
    first_controller = loaded.create_controller()
    second_controller = loaded.create_controller()

    assert loaded.frame_size == (96, 64)
    assert loaded.profile_id == "menu-test-v1"
    assert loaded.source_run_ids == ("cal-01",)
    assert loaded.template_provenance[0].template_id == "lobby-preparation"
    assert loaded.template_provenance[0].source_run_id == "cal-01"
    assert loaded.template_provenance[0].source_sequence == 0
    assert loaded.template_provenance[0].page_sha256 == hashlib.sha256(
        (tmp_path / "lobby-page.png").read_bytes()
    ).hexdigest()
    assert observation.scene is MenuScene.LOBBY
    assert observation.accepted is True
    assert observation.action_accepted is True
    assert observation.action_point == (55.0, 34.0)
    assert first_controller is not second_controller
    assert first_controller.step(
        observation,
        now_ns=observation.captured_at_ns,
    ).status is MenuControllerStatus.OBSERVING


def test_menu_profile_supports_post_match_scene(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)

    loaded = load_menu_profile(path)

    assert MenuScene.POST_MATCH in loaded.stop_scenes
    assert loaded.transitions[0].target is MenuScene.POST_MATCH


def test_menu_profile_rejects_template_path_escape(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["page"]["image"] = "../outside.png"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Profile 目录"):
        load_menu_profile(path)


def test_menu_profile_rejects_template_hash_mismatch(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["page"]["sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_menu_profile(path)


def test_menu_profile_rejects_unknown_source_run(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["source_run_id"] = "not-declared"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="source_run_id"):
        load_menu_profile(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("profile_id", "", "profile_id"),
        ("expected_frame_size", [True, 64], "expected_frame_size"),
        ("source_runs", [{"run_id": "cal-01", "split": "blind"}], "calibration"),
    ],
)
def test_menu_profile_rejects_invalid_root_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_menu_profile(path)


def test_menu_profile_rejects_action_without_region(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["action_region"] = None
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="action_region"):
        load_menu_profile(path)


@pytest.mark.parametrize("field", ["minimum_scene_margin", "detector_margin"])
def test_menu_profile_rejects_zero_ambiguity_margin(
    tmp_path: Path,
    field: str,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if field == "minimum_scene_margin":
        raw[field] = 0
    else:
        raw["templates"][0]["page"]["minimum_margin"] = 0
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="margin"):
        load_menu_profile(path)


def test_menu_profile_requires_strict_utf8(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_bytes(text.encode("utf-16"))

    with pytest.raises(ValueError, match="UTF-8"):
        load_menu_profile(path)


def test_menu_profile_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            '"schema_version": 1',
            '"schema_version": 999, "schema_version": 1',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复字段"):
        load_menu_profile(path)


@pytest.mark.parametrize("mutation", ["duplicate_split", "boolean_sequence"])
def test_menu_profile_rejects_ambiguous_or_coerced_source_manifest(
    tmp_path: Path,
    mutation: str,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / raw["source_runs"][0]["directory"] / "manifest.jsonl"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if mutation == "duplicate_split":
        lines[0] = lines[0].replace(
            '"dataset_split": "calibration"',
            '"dataset_split": "blind", "dataset_split": "calibration"',
            1,
        )
    else:
        first = json.loads(lines[0])
        first["sequence"] = False
        lines[0] = json.dumps(first)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw["source_runs"][0]["frame_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=r"重复字段|sequence"):
        load_menu_profile(path)


def test_menu_profile_rejects_float_source_resolution(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    run_path = tmp_path / raw["source_runs"][0]["directory"] / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["resolution"] = [96.0, 64.0]
    run_path.write_text(json.dumps(run), encoding="utf-8")
    raw["source_runs"][0]["run_json_sha256"] = hashlib.sha256(
        run_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="分辨率"):
        load_menu_profile(path)


def test_menu_profile_rejects_disconnected_transition_chain(tmp_path: Path) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["transitions"].append(dict(raw["transitions"][0]))
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="首尾连续"):
        load_menu_profile(path)


@pytest.mark.parametrize(
    ("field", "scene"),
    [
        ("source", "death_summary"),
        ("target", "in_match"),
        ("stop", "death_summary"),
    ],
)
def test_menu_profile_rejects_unobservable_transition_or_stop_scene(
    tmp_path: Path,
    field: str,
    scene: str,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if field == "stop":
        raw["stop_scenes"] = [scene]
    else:
        raw["transitions"][0][field] = scene
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="对应模板"):
        load_menu_profile(path)


def test_menu_profile_rejects_stop_scene_that_is_a_transition_source(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["stop_scenes"] = ["lobby"]
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=r"stop_scenes.*来源"):
        load_menu_profile(path)


def test_menu_profile_rejects_repeated_transition_token_in_cycle(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["transitions"].extend(
        [
            {
                "source": "post_match",
                "target": "lobby",
                "action_kind": "key",
                "key": "space",
            },
            dict(raw["transitions"][0]),
        ]
    )
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="重复"):
        load_menu_profile(path)


def test_menu_profile_rejects_same_page_and_action_image_content(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["action"]["image"] = raw["templates"][0]["page"]["image"]
    raw["templates"][0]["action"]["sha256"] = raw["templates"][0]["page"]["sha256"]
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="独立"):
        load_menu_profile(path)


def test_menu_profile_rejects_same_decoded_anchor_pixels_with_different_png(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    page_path = tmp_path / raw["templates"][0]["page"]["image"]
    page = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    encoded, payload = cv2.imencode(
        ".png",
        page,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    assert encoded
    action_path = tmp_path / "same-pixels-action.png"
    action_path.write_bytes(payload.tobytes())
    assert hashlib.sha256(action_path.read_bytes()).hexdigest() != raw["templates"][0][
        "page"
    ]["sha256"]
    raw["templates"][0]["action"]["image"] = action_path.name
    raw["templates"][0]["action"]["sha256"] = hashlib.sha256(
        action_path.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="独立"):
        load_menu_profile(path)


def test_menu_profile_rejects_disjoint_action_search_and_click_regions(
    tmp_path: Path,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["action"]["search_roi"] = {
        "left": 0,
        "top": 0,
        "width": 20,
        "height": 15,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=r"action_region.*search_roi"):
        load_menu_profile(path)


@pytest.mark.parametrize("scales", [[100.0], [1_000_000.0]])
def test_menu_profile_rejects_scales_that_cannot_fit_search_roi(
    tmp_path: Path,
    scales: list[float],
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["page"]["scales"] = scales
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=r"scale.*search_roi"):
        load_menu_profile(path)


@pytest.mark.parametrize("scales", [[0.000001], [1e308], [10**400]])
def test_menu_profile_rejects_degenerate_or_overflowing_scales_as_value_error(
    tmp_path: Path,
    scales: list[float],
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["templates"][0]["page"]["scales"] = scales
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="scale"):
        load_menu_profile(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["source_runs"][0].update(directory="missing"), "目录"),
        (
            lambda raw: raw["templates"][0].update(source_sequence=999999),
            "source_sequence",
        ),
        (
            lambda raw: raw["templates"][0].update(source_frame_sha256="0" * 64),
            "source_frame_sha256",
        ),
    ],
)
def test_menu_profile_rejects_unverifiable_source_provenance(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    path, _, _ = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutation(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_menu_profile(path)
