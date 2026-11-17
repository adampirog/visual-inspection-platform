from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

from visual_inspection_platform.postprocessors import Postprocessor, get_postprocessor_class
from visual_inspection_platform.types import AnomalyMaps, BinaryMasks, ImageBatch
from visual_inspection_platform.utils import generate_model_id

from .manifest import ModelManifest, TrainingMetadata


class AnomalyDetector(ABC):
    """
    Base interface for serializable anomaly detectors.

    Each instance defines its expected image_size as (height, width).
    Shared preprocessing is performed by the caller; model-specific
    preprocessing is the responsibility of the detector.
    """

    MODEL_FAMILY: ClassVar[str]
    ARTIFACT_VERSION: ClassVar[str]

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        model_id: str | None = None,
        training_metadata: TrainingMetadata | None = None,
        postprocessor: Postprocessor | None = None,
    ) -> None:
        if model_id is None:
            self.model_id = generate_model_id(model_family=type(self).MODEL_FAMILY)
        else:
            self.model_id = model_id

        self.image_size = image_size
        self.training_metadata = training_metadata
        self.postprocessor = postprocessor

    @abstractmethod
    def predict(self, images: ImageBatch) -> AnomalyMaps:
        """
        Predict pixel-level anomaly scores.

        Inputs are RGB float32 arrays of shape [N, 3, H, W] in [0, 1].
        Outputs are float32 arrays of shape [N, H, W].
        Higher scores indicate stronger anomalies.
        Implementations must not modify the input batch.
        """

    def predict_mask(self, maps: AnomalyMaps) -> BinaryMasks:
        """Convert precomputed anomaly maps to masks using the configured postprocessor."""
        if self.postprocessor is None:
            raise RuntimeError("No postprocessor configured")

        return self.postprocessor(maps)

    @abstractmethod
    def _save(self, path: Path) -> None:
        """Save model-specific state; common metadata is handled by the base class."""

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to disk."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._save(path)

        if self.postprocessor is not None:
            postprocessor_manifest = self.postprocessor.save(path / "postprocessor")
        else:
            postprocessor_manifest = None

        ModelManifest(
            model_id=self.model_id,
            image_size=self.image_size,
            training_metadata=self.training_metadata,
            postprocessor=postprocessor_manifest,
            model_family=self.MODEL_FAMILY,
            artifact_version=self.ARTIFACT_VERSION,
        ).save(path / "MANIFEST.json")

    @classmethod
    @abstractmethod
    def _load(cls, path: Path, manifest: ModelManifest) -> Self:
        """Restore model-specific state using the identity and image size from the manifest."""

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a serialized model from disk."""

        path = Path(path)
        manifest = ModelManifest.load(path / "MANIFEST.json")

        if manifest.model_family != cls.MODEL_FAMILY:
            raise ValueError(
                f"Model family mismatch: expected '{cls.MODEL_FAMILY}', "
                f"got '{manifest.model_family}'"
            )

        model = cls._load(path, manifest=manifest)

        if model.model_id != manifest.model_id:
            raise RuntimeError("Loaded model ID does not match manifest")

        model.training_metadata = manifest.training_metadata

        if manifest.postprocessor is not None:
            postprocessor = get_postprocessor_class(manifest.postprocessor.kind).load(
                path=path / "postprocessor", manifest=manifest.postprocessor
            )
            model.postprocessor = postprocessor
        else:
            model.postprocessor = None

        return model
