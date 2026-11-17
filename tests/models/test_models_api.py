from pathlib import Path

import numpy as np
import pytest

from visual_inspection_platform.models import AnomalyDetector, ModelManifest, TrainingMetadata
from visual_inspection_platform.postprocessors import ThresholdPostprocessor
from visual_inspection_platform.postprocessors.registry import POSTPROCESSORS


class DummyDetector(AnomalyDetector):
    MODEL_FAMILY = "dummy"
    ARTIFACT_VERSION = "1"

    def __init__(self, bias: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.bias = bias

    def predict(self, images):
        return images[:, 0] + self.bias

    def _save(self, path: Path) -> None:
        (path / "bias.txt").write_text(str(self.bias), encoding="utf-8")

    @classmethod
    def _load(cls, path: Path, manifest: ModelManifest):
        return cls(
            bias=float((path / "bias.txt").read_text(encoding="utf-8")),
            model_id=manifest.model_id,
            training_metadata=manifest.training_metadata,
            postprocessor=ThresholdPostprocessor(0.9),
            image_size=manifest.image_size,
        )


def test_model_roundtrip(tmp_path):
    metadata = TrainingMetadata(trainer="dummy_trainer", config={"parameter": 42})
    model = DummyDetector(
        bias=0.25, model_id="dummy-001", training_metadata=metadata, image_size=(64, 64)
    )
    model.save(tmp_path)

    manifest = ModelManifest.load(tmp_path / "MANIFEST.json")
    assert manifest.model_id == "dummy-001"
    assert manifest.model_family == "dummy"
    assert manifest.training_metadata == metadata
    assert manifest.postprocessor is None

    loaded = DummyDetector.load(tmp_path)
    assert loaded.model_id == model.model_id
    assert loaded.training_metadata == metadata
    assert loaded.bias == 0.25
    assert loaded.postprocessor is None

    images = np.zeros((2, 3, 4, 4), dtype=np.float32)
    np.testing.assert_array_equal(
        loaded.predict(images),
        np.full((2, 4, 4), 0.25, dtype=np.float32),
    )


def test_model_postprocessor_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setitem(POSTPROCESSORS, ThresholdPostprocessor.KIND, ThresholdPostprocessor)

    model = DummyDetector(postprocessor=ThresholdPostprocessor(0.5), image_size=(64, 64))
    model.save(tmp_path)

    loaded = DummyDetector.load(tmp_path)
    assert isinstance(loaded.postprocessor, ThresholdPostprocessor)
    assert loaded.postprocessor.threshold == 0.5

    maps = np.array([[[0.2, 0.8]]], dtype=np.float32)
    np.testing.assert_array_equal(
        loaded.predict_mask(maps),
        np.array([[[False, True]]]),
    )


def test_missing_postprocessor():
    model = DummyDetector(image_size=(64, 64))

    with pytest.raises(RuntimeError, match="No postprocessor configured"):
        model.predict_mask(np.zeros((1, 4, 4), dtype=np.float32))


def test_model_family_mismatch(tmp_path):
    DummyDetector(image_size=(64, 64)).save(tmp_path)

    class OtherDetector(DummyDetector):
        MODEL_FAMILY = "other"

    with pytest.raises(ValueError, match="Model family mismatch"):
        OtherDetector.load(tmp_path)


def test_model_id_mismatch(tmp_path):
    DummyDetector(image_size=(64, 64)).save(tmp_path)

    class InvalidDetector(DummyDetector):
        @classmethod
        def _load(cls, path: Path, manifest: ModelManifest):
            return cls(model_id="incorrect", image_size=manifest.image_size)

    with pytest.raises(RuntimeError, match="Loaded model ID"):
        InvalidDetector.load(tmp_path)
