from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, PositiveInt

from visual_inspection_platform.postprocessors import PostprocessorManifest
from visual_inspection_platform.version import __version__


MANIFEST_VERSION = "0.1.0"


class TrainingMetadata(BaseModel):
    """Fully qualified trainer name and its serialized configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trainer: str
    config: dict[str, JsonValue]


class ModelManifest(BaseModel):
    """Model metadata readable without loading model parameters.

    image_size is (height, width). artifact_version identifies the model-specific
    serialization format; manifest_version identifies the common metadata format.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_family: str
    image_size: tuple[PositiveInt, PositiveInt]
    postprocessor: PostprocessorManifest | None = None
    training_metadata: TrainingMetadata | None = None
    artifact_version: str
    package_version: str = __version__
    manifest_version: str = MANIFEST_VERSION

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate_json(handle.read())

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=2))
