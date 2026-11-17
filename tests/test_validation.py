import numpy as np
import pytest

from visual_inspection_platform.validation import validate_image_batch


def test_valid_image_batch():
    images = np.zeros((2, 3, 4, 6), dtype=np.float32)

    validate_image_batch(images)


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((3, 4, 6), dtype=np.float32),
        np.zeros((2, 1, 4, 6), dtype=np.float32),
        np.zeros((2, 3, 4, 6), dtype=np.float64),
        np.zeros((0, 3, 4, 6), dtype=np.float32),
    ],
)
def test_invalid_image_batch(images):
    with pytest.raises(ValueError):
        validate_image_batch(images)
