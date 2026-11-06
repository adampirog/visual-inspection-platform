from pathlib import Path

import numpy as np
import pytest

from visual_inspection_platform.models import (
    AnomalyDetector,
    AnomalyPrediction,
    InspectionRequest,
    ModelManifest,
    TrainingMetadata,
)


class DummyDetector(AnomalyDetector):
    MODEL_FAMILY = "dummy"
    ARTIFACT_VERSION = "1"

    def predict(self, request: InspectionRequest) -> AnomalyPrediction:
        return AnomalyPrediction(scores=np.zeros(len(request.images), dtype=np.float32))

    def _save(self, path: Path) -> None:
        (path / "state.txt").write_text("dummy", encoding="utf-8")

    @classmethod
    def _load(cls, path: Path, manifest: ModelManifest):
        assert (path / "state.txt").read_text(encoding="utf-8") == "dummy"
        return cls(model_id=manifest.model_id, training_metadata=manifest.training_metadata)


def test_model_roundtrip(tmp_path):
    metadata = TrainingMetadata(trainer="dummy_trainer", config={"parameter": 42})
    detector = DummyDetector(model_id="dummy-001", training_metadata=metadata)

    artifact = tmp_path / "model"
    detector.save(artifact)

    manifest = ModelManifest.load(artifact / "MANIFEST.json")
    assert manifest.model_id == detector.model_id
    assert manifest.model_family == "dummy"
    assert manifest.artifact_version == "1"
    assert manifest.training_metadata == metadata

    loaded = DummyDetector.load(artifact)
    assert loaded.model_id == detector.model_id

    request = InspectionRequest(images=np.zeros((2, 3, 8, 8), dtype=np.float32))
    prediction = loaded.predict(request)

    assert prediction.scores.shape == (2,)
    assert prediction.anomaly_maps is None


def test_model_family_mismatch(tmp_path):
    DummyDetector().save(tmp_path)

    class DifferentDetector(DummyDetector):
        MODEL_FAMILY = "different"

    with pytest.raises(ValueError, match="Cannot load saved model"):
        DifferentDetector.load(tmp_path)
