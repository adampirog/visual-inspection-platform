from pathlib import Path
from typing import Self, cast, override

import onnxruntime as ort

from visual_inspection_platform.models import (
    AnomalyDetector,
    ModelManifest,
    TrainingMetadata,
)
from visual_inspection_platform.postprocessors import Postprocessor
from visual_inspection_platform.types import AnomalyMaps, ImageBatch
from visual_inspection_platform.validation import validate_image_batch


class AutoencoderDetector(AnomalyDetector):
    """Reconstruction-based anomaly detector backed by ONNX Runtime.

    The ONNX graph computes pixel-level squared reconstruction errors
    averaged across color channels.
    """

    MODEL_FAMILY = "autoencoder"
    ARTIFACT_VERSION = "1"

    def __init__(
        self,
        onnx_model: bytes,
        image_size: tuple[int, int],
        *,
        model_id: str | None = None,
        training_metadata: TrainingMetadata | None = None,
        postprocessor: Postprocessor | None = None,
    ) -> None:
        super().__init__(
            image_size=image_size,
            model_id=model_id,
            training_metadata=training_metadata,
            postprocessor=postprocessor,
        )

        self._onnx_model = onnx_model
        self._session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

    @override
    def predict(self, images: ImageBatch) -> AnomalyMaps:
        validate_image_batch(images)

        if images.shape[2:] != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {images.shape[2:]}")

        maps = self._session.run(["anomaly_maps"], {"images": images})[0]

        return cast(AnomalyMaps, maps)

    @override
    def _save(self, path: Path) -> None:
        (path / "model.onnx").write_bytes(self._onnx_model)

    @classmethod
    @override
    def _load(cls, path: Path, manifest: ModelManifest) -> Self:
        return cls(
            onnx_model=(path / "model.onnx").read_bytes(),
            image_size=manifest.image_size,
            model_id=manifest.model_id,
        )
