import pytest

from visual_inspection_platform.models.pca import PCADetector
from visual_inspection_platform.serving.manager import ModelManager, ModelNotFoundError
from visual_inspection_platform.serving.registry import create_default_registry


def test_discover(model_path):
    manager = ModelManager(create_default_registry())

    assert manager.discover(model_path) == ("model-a", "model-b")
    assert manager.n_models == 2
    assert manager.n_loaded == 0

    models = {model.model_id: model for model in manager.available_models()}

    assert set(models) == {"model-a", "model-b"}
    assert models["model-a"].model_family == "pca"
    assert not models["model-a"].loaded

    assert manager.discover(model_path) == ()
    assert manager.n_models == 2

    assert models["model-a"].image_size == (2, 3)
    assert models["model-a"].has_postprocessor
    assert not models["model-b"].has_postprocessor


def test_model_lifecycle(manager):
    model = manager.load("model-a")

    assert isinstance(model, PCADetector)
    assert model.image_size == (2, 3)
    assert manager.is_loaded("model-a")
    assert manager.n_loaded == 1

    assert manager.load("model-a") is model

    manager.unload("model-a")

    assert not manager.is_loaded("model-a")
    assert manager.n_loaded == 0
    assert manager.n_models == 2

    reloaded = manager.load("model-a")

    assert reloaded is not model
    assert reloaded.model_id == model.model_id
    assert manager.is_loaded("model-a")


def test_unknown_model(manager):
    with pytest.raises(ModelNotFoundError):
        manager.load("unknown")


def test_discover_invalid_manifest(model_path):
    path = model_path / "invalid"
    path.mkdir()
    (path / "MANIFEST.json").write_text("{}", encoding="utf-8")

    manager = ModelManager(create_default_registry())

    assert manager.discover(model_path) == ("model-a", "model-b")
    assert manager.n_models == 2
