import numpy as np
import pytest

from delta_vision.perception import ColorAnchorDetector


def test_color_anchor_detector_returns_centroid_and_confidence() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[30:50, 70:90] = (20, 230, 20)
    detector = ColorAnchorDetector(
        label="player",
        bgr=(20, 230, 20),
        tolerance=5,
        minimum_area=300,
        confidence_threshold=0.9,
    )

    observation = detector.detect(image)

    assert observation.label == "player"
    assert observation.centroid == pytest.approx((79.5, 39.5))
    assert observation.area == 400
    assert observation.confidence == pytest.approx(1.0)
    assert observation.accepted is True


def test_color_anchor_detector_rejects_small_low_confidence_anchor() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[10:15, 10:15] = (20, 230, 20)
    detector = ColorAnchorDetector(
        label="player",
        bgr=(20, 230, 20),
        tolerance=5,
        minimum_area=100,
        confidence_threshold=0.9,
    )

    observation = detector.detect(image)

    assert observation.area == 25
    assert observation.confidence == pytest.approx(0.25)
    assert observation.centroid is None
    assert observation.candidate_centroid == pytest.approx((12.0, 12.0))
    assert observation.accepted is False


def test_color_anchor_detector_returns_uncertain_when_anchor_is_absent() -> None:
    detector = ColorAnchorDetector(
        label="player",
        bgr=(20, 230, 20),
        tolerance=5,
        minimum_area=100,
        confidence_threshold=0.9,
    )

    observation = detector.detect(np.zeros((40, 40, 3), dtype=np.uint8))

    assert observation.centroid is None
    assert observation.area == 0
    assert observation.confidence == 0
    assert observation.accepted is False


def test_color_anchor_detector_uses_largest_component() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[5:10, 5:10] = (20, 230, 20)
    image[30:50, 60:80] = (20, 230, 20)
    detector = ColorAnchorDetector(
        label="player",
        bgr=(20, 230, 20),
        tolerance=5,
        minimum_area=300,
        confidence_threshold=0.9,
    )

    observation = detector.detect(image)

    assert observation.area == 400
    assert observation.centroid == pytest.approx((69.5, 39.5))
    assert observation.accepted is True


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.float32),
    ],
)
def test_color_anchor_detector_rejects_invalid_image(image) -> None:
    detector = ColorAnchorDetector(
        label="player",
        bgr=(20, 230, 20),
        tolerance=5,
        minimum_area=10,
        confidence_threshold=0.9,
    )

    with pytest.raises(ValueError, match="uint8 BGR"):
        detector.detect(image)


def test_color_anchor_detector_rejects_zero_confidence_threshold() -> None:
    with pytest.raises(ValueError, match="置信度阈值"):
        ColorAnchorDetector(
            label="player",
            bgr=(20, 230, 20),
            tolerance=5,
            minimum_area=10,
            confidence_threshold=0,
        )
