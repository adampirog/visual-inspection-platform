import numpy as np
import pytest

from visual_inspection_platform.validation import (
    validate_image_batch,
    validate_prediction,
    validate_rgb_image,
)


def test_valid_arrays():
    validate_rgb_image(np.zeros((8, 8, 3), dtype=np.uint8))
    validate_image_batch(np.zeros((2, 3, 8, 8), dtype=np.float32))

    scores = np.zeros(2, dtype=np.float32)
    maps = np.zeros((2, 8, 8), dtype=np.float32)

    validate_prediction(scores, maps, batch_shape=(2, 3, 8, 8))
    validate_prediction(scores, None, batch_shape=(2, 3, 8, 8))


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (np.zeros((8, 8), dtype=np.uint8), ValueError),
        (np.zeros((8, 8, 3), dtype=np.float32), ValueError),
        (np.zeros((0, 8, 3), dtype=np.uint8), ValueError),
        (None, TypeError),
    ],
)
def test_invalid_rgb_image(image, error):
    with pytest.raises(error):
        validate_rgb_image(image)


@pytest.mark.parametrize(
    ("images", "error"),
    [
        (np.zeros((3, 8, 8), dtype=np.float32), ValueError),
        (np.zeros((2, 1, 8, 8), dtype=np.float32), ValueError),
        (np.zeros((2, 3, 8, 8), dtype=np.float64), ValueError),
        (np.zeros((0, 3, 8, 8), dtype=np.float32), ValueError),
        (None, TypeError),
    ],
)
def test_invalid_image_batch(images, error):
    with pytest.raises(error):
        validate_image_batch(images)


@pytest.mark.parametrize(
    ("scores", "maps"),
    [
        (np.zeros(1, dtype=np.float32), None),
        (np.zeros(2, dtype=np.float64), None),
        (np.zeros(2, dtype=np.float32), np.zeros((2, 4, 4), dtype=np.float32)),
    ],
)
def test_invalid_prediction(scores, maps):
    with pytest.raises(ValueError):
        validate_prediction(scores, maps, batch_shape=(2, 3, 8, 8))
