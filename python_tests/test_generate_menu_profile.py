import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from delta_vision.frames import DatasetContentDigest
from delta_vision.generate_menu_profile import generate_menu_profile
from delta_vision.menu_automation import MenuScene
from delta_vision.menu_profile import load_menu_profile

FRAME_SIZE = (1920, 1080)


def _region(left: int, top: int, width: int, height: int) -> dict[str, int]:
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


def _write_dataset(
    root: Path,
    *,
    run_id: str,
    seed: int,
    frame_size: tuple[int, int] = FRAME_SIZE,
) -> tuple[Path, np.ndarray]:
    dataset = root / f"dataset-{run_id}-{seed}"
    frames = dataset / "frames"
    frames.mkdir(parents=True)
    width, height = frame_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    random = np.random.default_rng(seed)
    image[10:50, 10:50] = random.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    image[100:140, 100:140] = random.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    image[200:240, 200:240] = random.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    image[300:340, 300:340] = random.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    image_path = frames / "frame-00000000.png"
    assert cv2.imwrite(str(image_path), image)
    manifest = {
        "captured_at_ns": 100_000_000,
        "height": height,
        "image": "frames/frame-00000000.png",
        "metadata": {
            "dataset_split": "calibration",
            "run_id": run_id,
        },
        "sequence": 0,
        "source": "profile-generator-test",
        "width": width,
    }
    (dataset / "manifest.jsonl").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dataset / "run.json").write_text(
        json.dumps(
            {
                "dataset_split": "calibration",
                "frame_count": 1,
                "resolution": [width, height],
                "run_id": run_id,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset, image


def _spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "menu-zero-cost-1080p-v1",
        "expected_frame_size": [1920, 1080],
        "minimum_scene_margin": 0.08,
        "detector_defaults": {
            "scales": [1.0],
            "score_threshold": 0.9,
            "minimum_margin": 0.08,
            "nms_radius_px": 8,
        },
        "controller": {
            "confirmation_frames": 3,
            "maximum_confirmation_span_ms": 750,
            "maximum_action_point_drift_px": 12,
            "maximum_page_point_drift_px": 12,
            "maximum_frame_age_ms": 250,
            "transition_timeout_ms": 8000,
        },
        "templates": [
            {
                "template_id": "base-start",
                "scene": "base",
                "source_run_id": "base-cal",
                "source_sequence": 0,
                "page": {
                    "crop": _region(10, 10, 40, 40),
                    "search_roi": _region(0, 0, 80, 80),
                },
                "action": {
                    "crop": _region(100, 100, 40, 40),
                    "search_roi": _region(80, 80, 100, 100),
                    "action_region": _region(90, 90, 80, 80),
                },
            },
            {
                "template_id": "active-match",
                "scene": "in_match",
                "source_run_id": "match-cal",
                "source_sequence": 0,
                "page": {
                    "crop": _region(200, 200, 40, 40),
                    "search_roi": _region(180, 180, 100, 100),
                },
            },
            {
                "template_id": "post-match-pass",
                "scene": "post_match",
                "source_run_id": "post-cal",
                "source_sequence": 0,
                "page": {
                    "crop": _region(300, 300, 40, 40),
                    "search_roi": _region(280, 280, 100, 100),
                },
            },
        ],
        "transitions": [
            {
                "source": "base",
                "target": "in_match",
                "action_kind": "click",
                "key": None,
            }
        ],
        "stop_scenes": ["in_match"],
    }


def _write_spec(path: Path, spec: dict[str, object] | None = None) -> Path:
    path.write_text(
        json.dumps(_spec() if spec is None else spec, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_generate_menu_profile_builds_loadable_multi_dataset_bundle(
    tmp_path: Path,
) -> None:
    base, base_image = _write_dataset(tmp_path, run_id="base-cal", seed=1)
    match, _ = _write_dataset(tmp_path, run_id="match-cal", seed=2)
    post, _ = _write_dataset(tmp_path, run_id="post-cal", seed=3)
    output = tmp_path / "menu-profile"

    result = generate_menu_profile(
        spec_path=_write_spec(tmp_path / "spec.json"),
        dataset_directories=[post, base, match],
        output_directory=output,
    )

    loaded = load_menu_profile(result.profile_path)
    assert result.profile_path == output / "menu.json"
    assert result.profile_id == "menu-zero-cost-1080p-v1"
    assert result.frame_size == FRAME_SIZE
    assert result.source_run_ids == ("base-cal", "match-cal", "post-cal")
    assert result.template_count == 3
    assert result.template_asset_count == 4
    assert loaded.frame_size == FRAME_SIZE
    assert loaded.stop_scenes == frozenset({MenuScene.IN_MATCH})
    assert loaded.source_run_ids == result.source_run_ids
    page = cv2.imread(str(output / "templates/base-start-page.png"))
    action = cv2.imread(str(output / "templates/base-start-action.png"))
    assert np.array_equal(page, base_image[10:50, 10:50])
    assert np.array_equal(action, base_image[100:140, 100:140])
    assert (
        loaded.template_provenance[0].page_sha256
        == hashlib.sha256((output / "templates/base-start-page.png").read_bytes()).hexdigest()
    )
    digest = DatasetContentDigest()
    digest.update(0, base_image)
    raw_profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    base_run = next(item for item in raw_profile["source_runs"] if item["run_id"] == "base-cal")
    assert base_run["dataset_content_sha256"] == digest.hexdigest()


def test_generate_menu_profile_rejects_non_1080p_dataset_before_output(
    tmp_path: Path,
) -> None:
    base, _ = _write_dataset(
        tmp_path,
        run_id="base-cal",
        seed=1,
        frame_size=(1280, 720),
    )
    match, _ = _write_dataset(tmp_path, run_id="match-cal", seed=2)
    post, _ = _write_dataset(tmp_path, run_id="post-cal", seed=3)
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match=r"1920x1080|分辨率"):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json"),
            dataset_directories=[base, match, post],
            output_directory=output,
        )

    assert not output.exists()


def test_generate_menu_profile_preserves_existing_output(tmp_path: Path) -> None:
    datasets = [
        _write_dataset(tmp_path, run_id=run_id, seed=seed)[0]
        for seed, run_id in enumerate(("base-cal", "match-cal", "post-cal"), start=1)
    ]
    output = tmp_path / "menu-profile"
    output.mkdir()
    marker = output / "owned-by-user.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="已经存在"):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json"),
            dataset_directories=datasets,
            output_directory=output,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda spec: spec["templates"][0].update(source_sequence=9),
            "source_sequence|序号",
        ),
        (
            lambda spec: spec["templates"][0]["page"].update(crop=_region(1900, 1060, 40, 40)),
            "crop|超出",
        ),
        (
            lambda spec: spec["templates"][0].pop("action"),
            "点击|action",
        ),
    ],
)
def test_generate_menu_profile_rejects_invalid_template_spec(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    datasets = [
        _write_dataset(tmp_path, run_id=run_id, seed=seed)[0]
        for seed, run_id in enumerate(("base-cal", "match-cal", "post-cal"), start=1)
    ]
    spec = _spec()
    mutate(spec)
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match=message):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json", spec),
            dataset_directories=datasets,
            output_directory=output,
        )

    assert not output.exists()


def test_generate_menu_profile_rejects_duplicate_dataset_run_id(
    tmp_path: Path,
) -> None:
    first, _ = _write_dataset(tmp_path, run_id="base-cal", seed=1)
    duplicate, _ = _write_dataset(tmp_path, run_id="base-cal", seed=2)
    match, _ = _write_dataset(tmp_path, run_id="match-cal", seed=3)
    post, _ = _write_dataset(tmp_path, run_id="post-cal", seed=4)
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match=r"重复.*run_id|run_id.*重复"):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json"),
            dataset_directories=[first, duplicate, match, post],
            output_directory=output,
        )

    assert not output.exists()


def test_generate_menu_profile_rejects_duplicate_spec_field_before_output(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match="重复字段"):
        generate_menu_profile(
            spec_path=spec_path,
            dataset_directories=[],
            output_directory=output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda spec: spec["controller"].update(typo_timeout_ms=100),
        lambda spec: spec["transitions"][0].update(typo_action="click"),
    ],
)
def test_generate_menu_profile_rejects_unknown_nested_spec_field(
    tmp_path: Path,
    mutate,
) -> None:
    spec = _spec()
    mutate(spec)
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match=r"未知字段|extra"):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json", spec),
            dataset_directories=[],
            output_directory=output,
        )

    assert not output.exists()


def test_generate_menu_profile_rejects_textureless_crop(tmp_path: Path) -> None:
    datasets = [
        _write_dataset(tmp_path, run_id=run_id, seed=seed)[0]
        for seed, run_id in enumerate(("base-cal", "match-cal", "post-cal"), start=1)
    ]
    base_frame = datasets[0] / "frames/frame-00000000.png"
    image = cv2.imread(str(base_frame))
    image[10:50, 10:50] = 0
    assert cv2.imwrite(str(base_frame), image)
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match="纹理不足"):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json"),
            dataset_directories=datasets,
            output_directory=output,
        )

    assert not output.exists()


def test_generate_menu_profile_rejects_identical_page_and_action_crop(
    tmp_path: Path,
) -> None:
    datasets = [
        _write_dataset(tmp_path, run_id=run_id, seed=seed)[0]
        for seed, run_id in enumerate(("base-cal", "match-cal", "post-cal"), start=1)
    ]
    spec = _spec()
    spec["templates"][0]["action"] = {
        "crop": _region(10, 10, 40, 40),
        "search_roi": _region(0, 0, 80, 80),
        "action_region": _region(0, 0, 80, 80),
    }
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match="独立图像"):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json", spec),
            dataset_directories=datasets,
            output_directory=output,
        )

    assert not output.exists()


def test_generate_menu_profile_rejects_page_only_template_in_click_source_scene(
    tmp_path: Path,
) -> None:
    datasets = [
        _write_dataset(tmp_path, run_id=run_id, seed=seed)[0]
        for seed, run_id in enumerate(("base-cal", "match-cal", "post-cal"), start=1)
    ]
    spec = _spec()
    spec["templates"].append(
        {
            "template_id": "base-page-only",
            "scene": "base",
            "source_run_id": "base-cal",
            "source_sequence": 0,
            "page": {
                "crop": _region(200, 200, 40, 40),
                "search_roi": _region(180, 180, 100, 100),
            },
        }
    )
    output = tmp_path / "menu-profile"

    with pytest.raises(ValueError, match=r"每个模板.*action"):
        generate_menu_profile(
            spec_path=_write_spec(tmp_path / "spec.json", spec),
            dataset_directories=datasets,
            output_directory=output,
        )

    assert not output.exists()


def test_generate_menu_profile_is_deterministic(tmp_path: Path) -> None:
    datasets = [
        _write_dataset(tmp_path, run_id=run_id, seed=seed)[0]
        for seed, run_id in enumerate(("base-cal", "match-cal", "post-cal"), start=1)
    ]
    spec_path = _write_spec(tmp_path / "spec.json")

    first = generate_menu_profile(
        spec_path=spec_path,
        dataset_directories=datasets,
        output_directory=tmp_path / "first",
    )
    second = generate_menu_profile(
        spec_path=spec_path,
        dataset_directories=reversed(datasets),
        output_directory=tmp_path / "second",
    )

    assert first.profile_path.read_bytes() == second.profile_path.read_bytes()
    assert (tmp_path / "first/templates/base-start-page.png").read_bytes() == (
        tmp_path / "second/templates/base-start-page.png"
    ).read_bytes()


def test_generate_menu_profile_concurrent_publish_has_single_winner(
    tmp_path: Path,
) -> None:
    datasets = [
        _write_dataset(tmp_path, run_id=run_id, seed=seed)[0]
        for seed, run_id in enumerate(("base-cal", "match-cal", "post-cal"), start=1)
    ]
    spec_path = _write_spec(tmp_path / "spec.json")
    output = tmp_path / "menu-profile"

    def generate() -> object:
        try:
            return generate_menu_profile(
                spec_path=spec_path,
                dataset_directories=datasets,
                output_directory=output,
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: generate(), range(2)))

    successes = [result for result in results if not isinstance(result, BaseException)]
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "exist" in str(failures[0]).lower() or "存在" in str(failures[0])
    assert load_menu_profile(output / "menu.json").profile_id == "menu-zero-cost-1080p-v1"


def test_checked_in_1080p_example_spec_builds_a_profile(tmp_path: Path) -> None:
    example_spec = Path(__file__).resolve().parents[1] / "configs/menu-1920.spec.example.json"
    datasets = []
    definitions = (
        (
            "homepage-windowed-1080p-home-clean-20260719-01",
            ((1460, 800, 370, 140), (1665, 949, 150, 32)),
        ),
        ("inmatch-active-1080p-20260719-01", ((820, 160, 280, 48),)),
        ("post-match-season-pass-1080p-20260719-01", ((800, 94, 320, 52),)),
    )
    for seed, (run_id, crops) in enumerate(definitions, start=10):
        dataset, _ = _write_dataset(tmp_path, run_id=run_id, seed=seed)
        image_path = dataset / "frames/frame-00000000.png"
        image = cv2.imread(str(image_path))
        random = np.random.default_rng(seed)
        for left, top, width, height in crops:
            image[top : top + height, left : left + width] = random.integers(
                0,
                256,
                (height, width, 3),
                dtype=np.uint8,
            )
        assert cv2.imwrite(str(image_path), image)
        datasets.append(dataset)

    result = generate_menu_profile(
        spec_path=example_spec,
        dataset_directories=datasets,
        output_directory=tmp_path / "example-output",
    )

    loaded = load_menu_profile(result.profile_path)
    assert loaded.profile_id == "menu-zero-cost-1080p-v1"
    assert loaded.frame_size == FRAME_SIZE
    assert len(loaded.template_provenance) == 3
