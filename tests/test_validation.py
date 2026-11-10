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
    validate_prediction(np.zeros((2, 8, 8), dtype=np.float32), batch_shape=(2, 3, 8, 8))


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (None, TypeError),
        (np.zeros((8, 8), dtype=np.uint8), ValueError),
        (np.zeros((8, 8, 1), dtype=np.uint8), ValueError),
        (np.zeros((8, 8, 3), dtype=np.float32), ValueError),
        (np.zeros((0, 8, 3), dtype=np.uint8), ValueError),
    ],
)
def test_invalid_rgb_image(image, error):
    with pytest.raises(error):
        validate_rgb_image(image)


@pytest.mark.parametrize(
    ("images", "error"),
    [
        (None, TypeError),
        (np.zeros((3, 8, 8), dtype=np.float32), ValueError),
        (np.zeros((2, 1, 8, 8), dtype=np.float32), ValueError),
        (np.zeros((2, 3, 8, 8), dtype=np.float64), ValueError),
        (np.zeros((0, 3, 8, 8), dtype=np.float32), ValueError),
        (np.zeros((2, 3, 0, 8), dtype=np.float32), ValueError),
    ],
)
def test_invalid_image_batch(images, error):
    with pytest.raises(error):
        validate_image_batch(images)


@pytest.mark.parametrize(
    ("maps", "error"),
    [
        (None, TypeError),
        (np.zeros((8, 8), dtype=np.float32), ValueError),
        (np.zeros((1, 8, 8), dtype=np.float32), ValueError),
        (np.zeros((2, 4, 4), dtype=np.float32), ValueError),
        (np.zeros((2, 8, 8), dtype=np.float64), ValueError),
    ],
)
def test_invalid_prediction(maps, error):
    with pytest.raises(error):
        validate_prediction(maps, batch_shape=(2, 3, 8, 8))
