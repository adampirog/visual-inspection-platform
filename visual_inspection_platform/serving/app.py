"""FastAPI application for serving serialized visual anomaly detectors."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL.Image import DecompressionBombError

from visual_inspection_platform.models import AnomalyDetector
from visual_inspection_platform.utils.images import load_images
from visual_inspection_platform.version import __version__

from .manager import ModelManager, ModelNotFoundError
from .registry import create_default_registry
from .schema import (
    HealthResponse,
    ModelLoadResponse,
    ModelRefreshResponse,
    ModelResponse,
    PredictionResponse,
)


API_VERSION = "0.1.0"


def _load_model(manager: ModelManager, model_id: str) -> AnomalyDetector:
    """Load a model or return HTTP 404 if it is unknown."""
    try:
        return manager.load(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found.") from exc


def create_app(manager: ModelManager, model_path: Path) -> FastAPI:
    """Create the serving API using an existing model manager.

    model_path is scanned by refresh operations, not during app creation.
    """

    app = FastAPI(title="Visual Inspection Platform", version=API_VERSION)
    model_path = model_path.resolve()

    @app.get("/health")
    def health() -> HealthResponse:
        """Report degraded status when no model artifacts are registered."""
        return HealthResponse(
            status="ok" if manager.n_models else "degraded",
            version=__version__,
            models_available=manager.n_models,
            models_loaded=manager.n_loaded,
        )

    @app.get("/v1/models")
    def models() -> list[ModelResponse]:
        return [
            ModelResponse(
                model_id=model.model_id,
                model_family=model.model_family,
                loaded=model.loaded,
                image_size=model.image_size,
                has_postprocessor=model.has_postprocessor,
            )
            for model in manager.available_models()
        ]

    @app.post("/v1/models/{model_id}/predictions")
    def predict(
        model_id: str,
        files: Annotated[list[UploadFile], File(min_length=1, max_length=32)],
        include_masks: bool = False,
    ) -> PredictionResponse:
        """Predict anomaly maps for uploaded images in request order.

        Images are resized to the model's input resolution. Scores are returned
        at that resolution, without normalization. Requesting masks without a
        configured postprocessor returns HTTP 409.
        """
        model = _load_model(manager, model_id)

        if include_masks and model.postprocessor is None:
            raise HTTPException(status_code=409, detail="No postprocessor configured.")

        try:
            images = load_images([upload.file for upload in files], size=model.image_size)
        except (OSError, DecompressionBombError) as exc:
            raise HTTPException(
                status_code=422, detail="Uploaded images could not be decoded."
            ) from exc

        maps = model.predict(images)
        masks = model.predict_mask(maps) if include_masks else None

        return PredictionResponse(
            model_id=model_id,
            filenames=[upload.filename or "" for upload in files],
            anomaly_maps=maps.tolist(),
            masks=masks.tolist() if masks is not None else None,
        )

    @app.post("/admin/models/refresh")
    def refresh_models() -> ModelRefreshResponse:
        """Discover new artifacts without replacing previously registered models."""
        return ModelRefreshResponse(added=list(manager.discover(model_path)))

    @app.post("/admin/models/{model_id}/load")
    def load_model(model_id: str) -> ModelLoadResponse:
        _load_model(manager, model_id)

        return ModelLoadResponse(model_id=model_id, loaded=True)

    @app.delete("/admin/models/{model_id}/load")
    def unload_model(model_id: str) -> ModelLoadResponse:
        """Release cached model state without removing its registration.

        Unknown model IDs are treated as already unloaded.
        """
        manager.unload(model_id)

        return ModelLoadResponse(model_id=model_id, loaded=False)

    return app


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)

    parser.add_argument("model_path", type=Path, help="Root directory containing model artifacts.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manager = ModelManager(create_default_registry())
    manager.discover(args.model_path)

    app = create_app(manager=manager, model_path=args.model_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
