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


class CAEDetector(AnomalyDetector):
    """Detect anomalies using channel-averaged squared reconstruction errors.

    onnx_model must contain a self-contained graph with input images and
    output anomaly_maps, named accordingly. Inference runs on CPU.
    """

    MODEL_FAMILY = "convolutional_autoencoder"
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

        if not any(
            output.name == "anomaly_maps"
            and output.type == "tensor(float)"
            and output.shape[1:] == list(image_size)
            for output in self._session.get_outputs()
        ):
            raise ValueError("ONNX graph must return float32 anomaly maps matching image_size")

    @override
    def predict(self, images: ImageBatch) -> AnomalyMaps:
        if images.shape[2:] != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {images.shape[2:]}")

        if images.shape[0] == 0:
            raise ValueError("Batch must be nonempty")

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
