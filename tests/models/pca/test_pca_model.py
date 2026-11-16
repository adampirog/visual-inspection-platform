import numpy as np
import pytest

from visual_inspection_platform.models.pca import PCADetector


@pytest.fixture
def model():
    return PCADetector(
        mean=np.zeros(12, dtype=np.float32),
        components=np.eye(1, 12, dtype=np.float32),
        image_size=(2, 2),
        model_id="pca-test",
    )


def test_predict(model):
    images = np.zeros((2, 3, 2, 2), dtype=np.float32)
    images[1, 0, 0, 0] = 5.0  # Reconstructed perfectly
    images[1, 1, 0, 0] = 2.0  # Reconstruction error

    maps = model.predict(images)

    assert maps.shape == (2, 2, 2)
    assert maps.dtype == np.float32
    np.testing.assert_array_equal(maps[0], 0.0)
    np.testing.assert_allclose(maps[1, 0, 0], 4 / 3)


def test_save_load(model, tmp_path):
    images = np.random.default_rng(42).random((2, 3, 2, 2), dtype=np.float32)

    model.save(tmp_path)
    loaded = PCADetector.load(tmp_path)

    assert loaded.model_id == model.model_id
    assert loaded.image_size == model.image_size
    np.testing.assert_array_equal(loaded.mean, model.mean)
    np.testing.assert_array_equal(loaded.components, model.components)
    np.testing.assert_array_equal(loaded.predict(images), model.predict(images))


def test_invalid_image_size(model):
    images = np.zeros((1, 3, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected image size"):
        model.predict(images)
