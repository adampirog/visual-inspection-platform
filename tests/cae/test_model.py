import numpy as np
import pytest
import torch

from visual_inspection_platform.models import ModelManifest
from visual_inspection_platform.models.cae import CAEDetector


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_predict(detector, network, image_batch, batch_size):
    images = image_batch[:batch_size].copy()
    original = images.copy()

    with torch.inference_mode():
        reconstruction = network(torch.from_numpy(images)).numpy()

    expected = np.square(images - reconstruction).mean(axis=1)
    actual = detector.predict(images)

    assert actual.shape == (batch_size, *detector.image_size)
    assert actual.dtype == np.float32

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(images, original)


def test_model_roundtrip(detector, image_batch, tmp_path):
    detector.save(tmp_path)

    manifest = ModelManifest.load(tmp_path / "MANIFEST.json")
    loaded = CAEDetector.load(tmp_path)

    assert manifest.model_family == CAEDetector.MODEL_FAMILY
    assert manifest.artifact_version == CAEDetector.ARTIFACT_VERSION

    assert loaded.model_id == detector.model_id
    assert loaded.image_size == detector.image_size
    assert loaded.training_metadata == detector.training_metadata

    expected = detector.predict(image_batch)
    actual = loaded.predict(image_batch)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(loaded.predict_mask(actual), actual > 0.1)


def test_invalid_image_size(detector):
    images = np.zeros((1, 3, 64, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected image size"):
        detector.predict(images)
