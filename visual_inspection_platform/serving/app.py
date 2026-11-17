"""FastAPI application for serving serialized visual anomaly detectors."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from visual_inspection_platform.data import preprocess_image
from visual_inspection_platform.version import __version__

from .manager import ModelManager
from .registry import create_default_registry
from .schema import (
    HealthResponse,
    ModelLoadResponse,
    ModelRefreshResponse,
    ModelResponse,
    PredictionResponse,
)


API_VERSION = "0.1.0"


def create_app(manager: ModelManager, model_path: Path) -> FastAPI:
    """Create a serving application backed by a model-artifact directory."""
    app = FastAPI(title="Visual Inspection Platform", version=API_VERSION)
    model_path = model_path.resolve()

    @app.get("/health")
    def health() -> HealthResponse:
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
            )
            for model in manager.available_models()
        ]

    @app.post("/v1/models/{model_id}/predictions")
    def predict(
        model_id: str,
        files: Annotated[list[UploadFile], File(min_length=1, max_length=32)],
        include_masks: bool = False,
    ) -> PredictionResponse:
        try:
            model = manager.load(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found") from exc

        if include_masks and model.postprocessor is None:
            raise HTTPException(status_code=409, detail="No postprocessor configured")

        # Each registered serving model exposes its expected (height, width).
        height, width = getattr(model, "image_size")
        images = np.empty((len(files), 3, height, width), dtype=np.float32)

        for i, upload in enumerate(files):
            try:
                with Image.open(upload.file) as image:
                    images[i] = preprocess_image(image, (height, width))
            except (UnidentifiedImageError, OSError) as exc:
                raise HTTPException(
                    status_code=422, detail=f"Invalid image: {upload.filename}"
                ) from exc

        maps = model.predict(images)
        masks = model.predict_mask(maps).tolist() if include_masks else None

        return PredictionResponse(
            model_id=model_id,
            filenames=[upload.filename or "" for upload in files],
            anomaly_maps=maps.tolist(),
            masks=masks,
        )

    @app.post("/admin/models/refresh")
    def refresh_models() -> ModelRefreshResponse:
        return ModelRefreshResponse(added=list(manager.discover(model_path)))

    @app.post("/admin/models/{model_id}/load")
    def load_model(model_id: str) -> ModelLoadResponse:
        try:
            manager.load(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found") from exc
        return ModelLoadResponse(model_id=model_id, loaded=True)

    @app.delete("/admin/models/{model_id}/load")
    def unload_model(model_id: str) -> ModelLoadResponse:
        try:
            manager.unload(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found") from exc
        return ModelLoadResponse(model_id=model_id, loaded=False)

    return app


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Root directory of model artifacts")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = ModelManager(create_default_registry())
    manager.discover(args.model_path)
    uvicorn.run(create_app(manager, args.model_path), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
