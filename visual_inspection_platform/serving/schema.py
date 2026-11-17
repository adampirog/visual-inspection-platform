"""HTTP response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    filenames: list[str]
    anomaly_maps: list[list[list[float]]]
    masks: list[list[list[bool]]] | None = None


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    model_family: str
    loaded: bool
    image_size: tuple[int, int]
    has_postprocessor: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "degraded"]
    version: str
    models_available: int
    models_loaded: int


class ModelRefreshResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    added: list[str]


class ModelLoadResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    loaded: bool
