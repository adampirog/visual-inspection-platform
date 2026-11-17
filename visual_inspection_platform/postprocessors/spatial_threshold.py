from pathlib import Path
from typing import Self

import numpy as np
from pydantic import JsonValue

from visual_inspection_platform.types import AnomalyMaps, BinaryMasks

from .base import Postprocessor


class SpatialThresholdPostprocessor(Postprocessor):
    """
    Apply location-specific thresholds with an additive offset.

    Thresholds are broadcast over the input batch.
    Spatial calibration assumes reasonably aligned images.
    """

    KIND = "spatial_threshold"
    VERSION = "1"

    def __init__(
        self,
        offset: float = 0.0,
        thresholds: np.ndarray | float = 0.0,
    ):
        self.offset = float(offset)
        self.thresholds = np.asarray(thresholds, dtype=np.float32)

    def __call__(self, maps: AnomalyMaps) -> BinaryMasks:
        return maps > self.thresholds + self.offset

    def get_params(self) -> dict[str, JsonValue]:
        return {"offset": self.offset}

    @classmethod
    def calibrate(cls, maps: AnomalyMaps, quantile: float = 0.995, offset: float = 0.0) -> Self:
        """Estimate a score quantile at each pixel location across normal images."""
        thresholds = np.quantile(maps, quantile, axis=0).astype(np.float32)
        return cls(offset=offset, thresholds=thresholds)

    def _save(self, path: Path) -> None:
        np.save(path / "thresholds.npy", self.thresholds)

    def _load(self, path: Path) -> None:
        self.thresholds = np.load(path / "thresholds.npy", allow_pickle=False)
