from pathlib import Path

import numpy as np
import pytest

from visual_inspection_platform.models import AnomalyDetector, ModelManifest, TrainingMetadata
from visual_inspection_platform.postprocessors import Postprocessor, PostprocessorManifest
from visual_inspection_platform.postprocessors.registry import POSTPROCESSORS


class DummyPostprocessor(Postprocessor):
    KIND = "dummy_threshold"
    VERSION = "1"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, maps):
        return maps >= self.threshold

    def get_params(self) -> dict:
        return {"threshold": self.threshold}


class StatefulPostprocessor(Postprocessor):
    KIND = "dummy_spatial"
    VERSION = "1"

    def __init__(self, thresholds: np.ndarray) -> None:
        self.thresholds = thresholds

    def __call__(self, maps):
        return maps >= self.thresholds

    def get_params(self):
        return {}

    def _save(self, path: Path) -> None:
        np.save(path / "thresholds.npy", self.thresholds)

    @classmethod
    def _load(cls, path: Path, manifest: PostprocessorManifest):
        return cls(np.load(path / "thresholds.npy", allow_pickle=False))


class DummyDetector(AnomalyDetector):
    MODEL_FAMILY = "dummy"
    ARTIFACT_VERSION = "1"

    def predict(self, images):
        return images[:, 0].copy()

    def _save(self, path: Path) -> None:
        (path / "state.txt").write_text("dummy", encoding="utf-8")

    @classmethod
    def _load(cls, path: Path, manifest: ModelManifest):
        assert (path / "state.txt").read_text(encoding="utf-8") == "dummy"
        return cls(
            model_id=manifest.model_id,
            training_metadata=manifest.training_metadata,
            postprocessor=DummyPostprocessor(0.9),
        )


def test_model_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setitem(POSTPROCESSORS, DummyPostprocessor.KIND, DummyPostprocessor)

    metadata = TrainingMetadata(trainer="dummy_trainer", config={"parameter": 42})
    model = DummyDetector(
        model_id="dummy-001",
        training_metadata=metadata,
        postprocessor=DummyPostprocessor(0.25),
    )
    model.save(tmp_path)

    manifest = ModelManifest.load(tmp_path / "MANIFEST.json")
    assert manifest.model_id == model.model_id
    assert manifest.model_family == model.MODEL_FAMILY
    assert manifest.training_metadata == metadata
    assert manifest.postprocessor.kind == DummyPostprocessor.KIND
    assert manifest.postprocessor.params == {"threshold": 0.25}

    loaded = DummyDetector.load(tmp_path)
    assert loaded.model_id == model.model_id
    assert loaded.training_metadata == metadata
    assert isinstance(loaded.postprocessor, DummyPostprocessor)
    assert loaded.postprocessor.threshold == 0.25

    images = np.zeros((2, 3, 4, 4), dtype=np.float32)
    images[0, 0, 1, 1] = 0.5

    maps = loaded.predict(images)
    assert maps.shape == (2, 4, 4)
    np.testing.assert_array_equal(loaded.predict_mask(maps), maps >= 0.25)


def test_model_without_postprocessor(tmp_path):
    DummyDetector().save(tmp_path)

    loaded = DummyDetector.load(tmp_path)
    assert loaded.postprocessor is None

    with pytest.raises(RuntimeError, match="No postprocessor configured"):
        loaded.predict_mask(np.zeros((1, 4, 4), dtype=np.float32))


def test_stateful_postprocessor_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setitem(POSTPROCESSORS, StatefulPostprocessor.KIND, StatefulPostprocessor)

    thresholds = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    model = DummyDetector(postprocessor=StatefulPostprocessor(thresholds))
    model.save(tmp_path)

    loaded = DummyDetector.load(tmp_path)
    assert isinstance(loaded.postprocessor, StatefulPostprocessor)

    np.testing.assert_array_equal(loaded.postprocessor.thresholds, thresholds)


def test_unknown_postprocessor(tmp_path, monkeypatch):
    monkeypatch.delitem(POSTPROCESSORS, DummyPostprocessor.KIND, raising=False)

    DummyDetector(postprocessor=DummyPostprocessor(0.5)).save(tmp_path)

    with pytest.raises(ValueError, match="not registered"):
        DummyDetector.load(tmp_path)


def test_model_family_mismatch(tmp_path):
    DummyDetector().save(tmp_path)

    class DifferentDetector(DummyDetector):
        MODEL_FAMILY = "different"

    with pytest.raises(ValueError, match="Model family mismatch"):
        DifferentDetector.load(tmp_path)


def test_postprocessor_kind_mismatch(tmp_path):
    manifest = PostprocessorManifest(kind="other", version="1")

    with pytest.raises(ValueError, match="Postprocessor kind mismatch"):
        DummyPostprocessor.load(tmp_path, manifest)
