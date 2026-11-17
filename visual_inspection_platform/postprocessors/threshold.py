from typing import Self

import numpy as np
from pydantic import JsonValue

from visual_inspection_platform.types import AnomalyMaps, BinaryMasks

from .base import Postprocessor


class ThresholdPostprocessor(Postprocessor):
    """Mark pixels whose anomaly scores strictly exceed a global threshold."""

    KIND = "threshold"
    VERSION = "1"

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def __call__(self, maps: AnomalyMaps) -> BinaryMasks:
        return maps > self.threshold

    def get_params(self) -> dict[str, JsonValue]:
        return {"threshold": self.threshold}

    @classmethod
    def calibrate(cls, maps: AnomalyMaps, quantile: float = 0.995) -> Self:
        """Estimate a global threshold from the quantile of all normal-image scores."""
        return cls(threshold=float(np.quantile(maps, quantile)))
