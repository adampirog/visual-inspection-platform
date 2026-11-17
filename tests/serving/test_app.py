from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from visual_inspection_platform.serving.app import create_app
from visual_inspection_platform.serving.manager import ModelManager
from visual_inspection_platform.serving.registry import create_default_registry


def _png(color: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (6, 4), color=color).save(buffer, format="PNG")

    return buffer.getvalue()


def test_health_and_models(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models_available"] == 2
    assert response.json()["models_loaded"] == 0

    response = client.get("/v1/models")

    assert response.status_code == 200

    models = {model["model_id"]: model for model in response.json()}

    assert set(models) == {"model-a", "model-b"}
    assert models["model-a"] == {
        "model_id": "model-a",
        "model_family": "pca",
        "loaded": False,
    }


def test_predictions(client, manager):
    response = client.post(
        "/v1/models/model-a/predictions",
        params={"include_masks": True},
        files=[
            ("files", ("black.png", _png(), "image/png")),
            ("files", ("red.png", _png((255, 0, 0)), "image/png")),
        ],
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_id"] == "model-a"
    assert payload["filenames"] == ["black.png", "red.png"]
    assert manager.is_loaded("model-a")

    expected = np.zeros((2, 2, 3), dtype=np.float32)
    expected[1] = 1 / 3
    expected[1, 0, 0] = 0.0

    np.testing.assert_allclose(payload["anomaly_maps"], expected)
    np.testing.assert_array_equal(payload["masks"], expected > 0.25)


def test_predictions_without_postprocessor(client):
    files = [("files", ("image.png", _png(), "image/png"))]

    response = client.post("/v1/models/model-b/predictions", files=files)

    assert response.status_code == 200
    assert response.json()["masks"] is None
    assert np.asarray(response.json()["anomaly_maps"]).shape == (1, 3, 4)

    response = client.post(
        "/v1/models/model-b/predictions",
        params={"include_masks": True},
        files=files,
    )

    assert response.status_code == 409


def test_unknown_model(client):
    response = client.post(
        "/v1/models/unknown/predictions",
        files=[("files", ("image.png", _png(), "image/png"))],
    )

    assert response.status_code == 404


def test_invalid_image(client):
    response = client.post(
        "/v1/models/model-a/predictions",
        files=[("files", ("invalid.png", b"not an image", "image/png"))],
    )

    assert response.status_code == 422


def test_admin_load_and_unload(client, manager):
    response = client.post("/admin/models/model-a/load")

    assert response.status_code == 200
    assert response.json() == {"model_id": "model-a", "loaded": True}
    assert manager.is_loaded("model-a")

    response = client.delete("/admin/models/model-a/load")

    assert response.status_code == 200
    assert response.json() == {"model_id": "model-a", "loaded": False}
    assert not manager.is_loaded("model-a")


def test_refresh_models(model_path):
    manager = ModelManager(create_default_registry())

    with TestClient(create_app(manager, model_path)) as client:
        assert client.get("/health").json()["status"] == "degraded"

        response = client.post("/admin/models/refresh")

        assert response.status_code == 200
        assert response.json() == {"added": ["model-a", "model-b"]}
        assert manager.n_models == 2
        assert manager.n_loaded == 0

        response = client.post("/admin/models/refresh")

        assert response.status_code == 200
        assert response.json() == {"added": []}
