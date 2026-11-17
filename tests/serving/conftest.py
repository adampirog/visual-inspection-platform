from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visual_inspection_platform.postprocessors import ThresholdPostprocessor
from visual_inspection_platform.serving.app import create_app
from visual_inspection_platform.serving.manager import ModelManager
from visual_inspection_platform.serving.registry import create_default_registry


@pytest.fixture
def model_path(tmp_path: Path, make_pca) -> Path:
    root = tmp_path / "models"

    make_pca(
        model_id="model-a",
        image_size=(2, 3),
        postprocessor=ThresholdPostprocessor(0.25),
    ).save(root / "model-a")

    make_pca(
        model_id="model-b",
        image_size=(3, 4),
    ).save(root / "model-b")

    return root


@pytest.fixture
def manager(model_path):
    manager = ModelManager(create_default_registry())
    manager.discover(model_path)

    return manager


@pytest.fixture
def client(manager, model_path):
    with TestClient(create_app(manager, model_path)) as client:
        yield client
