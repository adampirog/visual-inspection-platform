from pathlib import Path

import numpy as np
import pytest

from visual_inspection_platform.postprocessors import (
    Postprocessor,
    PostprocessorManifest,
    get_postprocessor_class,
)
from visual_inspection_platform.postprocessors.registry import POSTPROCESSORS


class ThresholdPostprocessor(Postprocessor):
    KIND = "threshold"
    VERSION = "1"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, maps):
        return maps >= self.threshold

    def get_params(self) -> dict:
        return {"threshold": self.threshold}


class SpatialPostprocessor(Postprocessor):
    KIND = "spatial"
    VERSION = "1"

    def __init__(self, offset: float, thresholds: np.ndarray | None = None):
        self.offset = offset
        self.thresholds = thresholds

    def __call__(self, maps):
        assert self.thresholds is not None
        return maps >= self.thresholds + self.offset

    def get_params(self) -> dict:
        return {"offset": self.offset}

    def _save(self, path: Path) -> None:
        assert self.thresholds is not None
        np.save(path / "thresholds.npy", self.thresholds)

    def _load(self, path: Path) -> None:
        self.thresholds = np.load(path / "thresholds.npy", allow_pickle=False)


def test_threshold_roundtrip(tmp_path):
    processor = ThresholdPostprocessor(0.4)
    manifest = processor.save(tmp_path)

    assert manifest.kind == "threshold"
    assert manifest.params == {"threshold": 0.4}

    loaded = ThresholdPostprocessor.load(tmp_path, manifest)

    maps = np.array([[[0.2, 0.5]]], dtype=np.float32)
    np.testing.assert_array_equal(loaded(maps), processor(maps))


def test_spatial_roundtrip(tmp_path):
    thresholds = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    processor = SpatialPostprocessor(offset=0.1, thresholds=thresholds)

    manifest = processor.save(tmp_path)
    loaded = SpatialPostprocessor.load(tmp_path, manifest)

    assert loaded.offset == 0.1
    np.testing.assert_array_equal(loaded.thresholds, thresholds)

    maps = np.array([[[0.2, 0.6], [0.8, 0.9]]], dtype=np.float32)
    np.testing.assert_array_equal(loaded(maps), processor(maps))


def test_postprocessor_kind_mismatch(tmp_path):
    manifest = PostprocessorManifest(kind="other", version="1")

    with pytest.raises(ValueError, match="Postprocessor kind mismatch"):
        ThresholdPostprocessor.load(tmp_path, manifest)


def test_registry(monkeypatch):
    monkeypatch.setitem(POSTPROCESSORS, ThresholdPostprocessor.KIND, ThresholdPostprocessor)

    assert get_postprocessor_class("threshold") is ThresholdPostprocessor

    with pytest.raises(ValueError, match="not registered"):
        get_postprocessor_class("unknown")
