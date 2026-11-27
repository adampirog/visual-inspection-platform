import numpy as np
import pytest
from fastapi.testclient import TestClient

from visual_inspection_platform.evaluation.evaluate_model import evaluate_model
from visual_inspection_platform.evaluation.metrics import (
    dice,
    pixel_ap,
    pixel_auroc,
    precision,
    recall,
)
from visual_inspection_platform.models.cae import CAEDetector
from visual_inspection_platform.serving.app import create_app
from visual_inspection_platform.serving.manager import ModelManager
from visual_inspection_platform.serving.registry import create_default_registry


def test_evaluate_model(detector, dataset_factory):
    data = dataset_factory(split="test", objects="cashew")

    images = data.load_images(detector.image_size)
    labels = data.load_masks(detector.image_size)

    maps = detector.predict(images)
    masks = detector.predict_mask(maps)

    result = evaluate_model(detector, data, batch_size=len(data))

    assert result["continuous"]["cashew"] == pytest.approx(
        {
            "pixel_auroc": pixel_auroc(maps, labels),
            "pixel_ap": pixel_ap(maps, labels),
        }
    )

    assert result["binary"]["cashew"] == pytest.approx(
        {
            "precision": precision(masks, labels),
            "recall": recall(masks, labels),
            "dice": dice(masks, labels),
        }
    )


def test_serving(detector, dataset_factory, tmp_path):
    detector.save(tmp_path / detector.model_id)

    manager = ModelManager(create_default_registry())

    assert manager.discover(tmp_path) == (detector.model_id,)
    assert not manager.is_loaded(detector.model_id)

    data = dataset_factory(split="test", objects="cashew")

    files = [
        ("files", (record.path.name, record.path.read_bytes(), "image/png")) for record in data
    ]

    with TestClient(create_app(manager, tmp_path)) as client:
        response = client.post(
            f"/v1/models/{detector.model_id}/predictions",
            params={"include_masks": True},
            files=files,
        )

    assert response.status_code == 200
    assert manager.is_loaded(detector.model_id)
    assert isinstance(manager.load(detector.model_id), CAEDetector)

    payload = response.json()

    assert payload["model_id"] == detector.model_id
    assert payload["filenames"] == [record.path.name for record in data]

    images = data.load_images(detector.image_size)
    expected = detector.predict(images)

    np.testing.assert_allclose(payload["anomaly_maps"], expected, rtol=1e-5, atol=1e-6)

    np.testing.assert_array_equal(payload["masks"], detector.predict_mask(expected))
