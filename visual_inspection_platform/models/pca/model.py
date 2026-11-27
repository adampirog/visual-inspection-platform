from pathlib import Path
from typing import Self

import numpy as np

from visual_inspection_platform.models import AnomalyDetector, ModelManifest, TrainingMetadata
from visual_inspection_platform.postprocessors import Postprocessor
from visual_inspection_platform.types import AnomalyMaps, ImageBatch
from visual_inspection_platform.validation import validate_image_batch


class PCADetector(AnomalyDetector):
    """Global PCA reconstruction detector operating on flattened CHW images.

    mean has shape [D] and components has shape [K, D], where
    D = 3 * height * width and K is the number of orthonormal components.
    """

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
            model_id=model_id,
            training_metadata=training_metadata,
            postprocessor=postprocessor,
            image_size=image_size,
        )

        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)

        if self.mean.shape != (3 * image_size[0] * image_size[1],):
            raise ValueError("PCA mean must match the flattened image size")

        if self.components.ndim != 2 or self.components.shape[1] != self.mean.size:
            raise ValueError("PCA components must have shape [K, D] matching the mean")

        if not np.isfinite(self.mean).all() or not np.isfinite(self.components).all():
            raise ValueError("PCA parameters must be finite")

    def predict(self, images: ImageBatch) -> AnomalyMaps:
        """Return squared reconstruction errors averaged across color channels."""

        validate_image_batch(images)

        n, c, h, w = images.shape

        if (h, w) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {(h, w)}")

        centered = images.reshape(n, -1) - self.mean
        centered -= (centered @ self.components.T) @ self.components

        np.square(centered, out=centered)

        return centered.reshape(n, c, h, w).mean(axis=1, dtype=np.float32)

    def _save(self, path: Path) -> None:
        np.savez(path / "pca.npz", mean=self.mean, components=self.components)

    @classmethod
    def _load(cls, path: Path, manifest: ModelManifest) -> Self:
        with np.load(path / "pca.npz", allow_pickle=False) as data:
            mean = data["mean"]
            components = data["components"]

        return cls(
            image_size=manifest.image_size,
            mean=mean,
            components=components,
            model_id=manifest.model_id,
        )
