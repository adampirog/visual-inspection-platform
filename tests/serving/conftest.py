from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from visual_inspection_platform.models.pca import PCADetector
from visual_inspection_platform.postprocessors import ThresholdPostprocessor
from visual_inspection_platform.serving.app import create_app
from visual_inspection_platform.serving.manager import ModelManager
from visual_inspection_platform.serving.registry import create_default_registry


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    root = tmp_path / "models"

    PCADetector(
        model_id="model-a",
        image_size=(2, 3),
        mean=np.zeros(18, dtype=np.float32),
        components=np.eye(1, 18, dtype=np.float32),
        postprocessor=ThresholdPostprocessor(0.25),
    ).save(root / "model-a")

    PCADetector(
        model_id="model-b",
        image_size=(3, 4),
        mean=np.zeros(36, dtype=np.float32),
        components=np.eye(1, 36, dtype=np.float32),
    ).save(root / "model-b")

    return root


@pytest.fixture
def manager(model_path: Path) -> ModelManager:
    manager = ModelManager(create_default_registry())
    manager.discover(model_path)

    return manager


@pytest.fixture
def client(manager: ModelManager, model_path: Path):
    with TestClient(create_app(manager, model_path)) as client:
        yield client
