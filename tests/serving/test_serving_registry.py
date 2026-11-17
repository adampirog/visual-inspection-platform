import pytest

from visual_inspection_platform.models.pca import PCADetector
from visual_inspection_platform.serving.registry import create_default_registry


def test_default_registry():
    registry = create_default_registry()

    assert registry["pca"] is PCADetector


def test_unknown_model_family():
    registry = create_default_registry()

    with pytest.raises(KeyError):
        registry["unknown"]
