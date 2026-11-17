import numpy as np
import pytest

from visual_inspection_platform.models import ModelManifest, TrainingMetadata
from visual_inspection_platform.models.pca import PCADetector
from visual_inspection_platform.postprocessors import ThresholdPostprocessor


def test_predict(make_pca):
    model = make_pca()

    images = np.zeros((2, 3, 2, 3), dtype=np.float32)
    images[1, 0, 0, 0] = 1.0
    images[1, 1, 0, 0] = 0.5

    original = images.copy()

    maps = model.predict(images)

    expected = np.zeros((2, 2, 3), dtype=np.float32)
    expected[1, 0, 0] = 0.25 / 3

    assert maps.dtype == np.float32
    np.testing.assert_allclose(maps, expected)
    np.testing.assert_array_equal(images, original)


@pytest.mark.parametrize(
    "image_size, threshold",
    [
        ((2, 3), None),
        ((3, 4), 0.1),
    ],
)
def test_model_roundtrip(make_pca, tmp_path, image_size, threshold):
    metadata = TrainingMetadata(
        trainer="test_trainer",
        config={"n_components": 1},
    )
    processor = None if threshold is None else ThresholdPostprocessor(threshold)

    model = make_pca(
        image_size=image_size,
        training_metadata=metadata,
        postprocessor=processor,
    )

    images = np.random.default_rng(42).random((2, 3, *image_size), dtype=np.float32)

    model.save(tmp_path)

    manifest = ModelManifest.load(tmp_path / "MANIFEST.json")
    loaded = PCADetector.load(tmp_path)

    assert manifest.image_size == image_size
    assert manifest.model_family == "pca"
    assert loaded.model_id == model.model_id
    assert loaded.image_size == image_size
    assert loaded.training_metadata == metadata

    expected = model.predict(images)
    actual = loaded.predict(images)

    np.testing.assert_array_equal(actual, expected)

    if processor is None:
        assert loaded.postprocessor is None
    else:
        np.testing.assert_array_equal(
            loaded.predict_mask(actual),
            expected > threshold,
        )


def test_invalid_image_size(make_pca):
    model = make_pca()
    images = np.zeros((1, 3, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected image size"):
        model.predict(images)


def test_missing_postprocessor(make_pca):
    model = make_pca()

    with pytest.raises(RuntimeError, match="No postprocessor configured"):
        model.predict_mask(np.zeros((1, 2, 3), dtype=np.float32))


def test_model_family_mismatch(make_pca, tmp_path):
    make_pca().save(tmp_path)

    class OtherDetector(PCADetector):
        MODEL_FAMILY = "other"

    with pytest.raises(ValueError, match="Model family mismatch"):
        OtherDetector.load(tmp_path)
