from pathlib import Path
from typing import Self

import numpy as np

from visual_inspection_platform.models import AnomalyDetector, ModelManifest, TrainingMetadata
from visual_inspection_platform.postprocessors import Postprocessor
from visual_inspection_platform.types import AnomalyMaps, ImageBatch
from visual_inspection_platform.validation import validate_image_batch


class PCADetector(AnomalyDetector):
    """Detect anomalies using PCA reconstruction error."""

    MODEL_FAMILY = "pca"
    ARTIFACT_VERSION = "1"

    def __init__(
        self,
        mean: np.ndarray,
        components: np.ndarray,
        image_size: tuple[int, int],
        *,
        model_id: str | None = None,
        training_metadata: TrainingMetadata | None = None,
        postprocessor: Postprocessor | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id, training_metadata=training_metadata, postprocessor=postprocessor
        )

        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)
        self.image_size = image_size

        height, width = image_size
        features = 3 * height * width

        if (
            height <= 0
            or width <= 0
            or self.mean.shape != (features,)
            or self.components.ndim != 2
            or self.components.shape[0] == 0
            or self.components.shape[1] != features
        ):
            raise ValueError("Invalid PCA parameters or image size")

    def predict(self, images: ImageBatch) -> AnomalyMaps:
        """Return per-pixel mean squared reconstruction error."""
        validate_image_batch(images)

        n, c, h, w = images.shape

        if (h, w) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {(h, w)}")

        centered = images.reshape(n, -1) - self.mean
        reconstructed = (centered @ self.components.T) @ self.components
        residual = (centered - reconstructed).reshape(n, c, h, w)

        return np.mean(residual**2, axis=1, dtype=np.float32)

    def _save(self, path: Path) -> None:
        np.savez(
            path / "pca.npz",
            mean=self.mean,
            components=self.components,
            image_size=self.image_size,
        )

    @classmethod
    def _load(cls, path: Path, manifest: ModelManifest) -> Self:
        with np.load(path / "pca.npz", allow_pickle=False) as data:
            mean = data["mean"]
            components = data["components"]
            size = data["image_size"]

        return cls(
            mean=mean,
            components=components,
            image_size=(int(size[0]), int(size[1])),
            model_id=manifest.model_id,
        )
