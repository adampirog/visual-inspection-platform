from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from visual_inspection_platform.types import AnomalyMaps, BinaryMasks


class PostprocessorManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    version: str
    params: dict[str, JsonValue] = Field(default_factory=dict)


class Postprocessor(ABC):
    """Base interface for serializable anomaly prediction models."""

    KIND: ClassVar[str]
    VERSION: ClassVar[str]

    @abstractmethod
    def __call__(self, maps: AnomalyMaps) -> BinaryMasks: ...

    @abstractmethod
    def get_params(self) -> dict[str, JsonValue]: ...

    @abstractmethod
    def _save(self, path: Path) -> None: ...

    def save(self, path: str | Path) -> PostprocessorManifest:
        """Serialize the fitted model to disk."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._save(path)

        return PostprocessorManifest(kind=self.KIND, version=self.VERSION, params=self.get_params())

    @classmethod
    @abstractmethod
    def _load(cls, path: Path, manifest: PostprocessorManifest) -> Self: ...

    @classmethod
    def load(cls, path: str | Path, manifest: PostprocessorManifest) -> Self:
        path = Path(path)

        if manifest.kind != cls.KIND:
            raise ValueError(
                "Cannot load saved postprocessor with "
                f"Manifest(kind={manifest.kind}) "
                f"with class {cls.__name__}(KIND={cls.KIND})."
            )

        return cls._load(path, manifest=manifest)
