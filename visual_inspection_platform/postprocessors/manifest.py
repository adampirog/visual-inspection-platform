from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from visual_inspection_platform.version import __version__


MANIFEST_VERSION = "0.1.0"


class PostprocessorManifest(BaseModel):
    """Metadata required to identify and load a serialized postprocessor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    model_family: str
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
