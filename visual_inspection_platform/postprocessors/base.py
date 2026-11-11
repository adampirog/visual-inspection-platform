from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from visual_inspection_platform.types import AnomalyMaps, BinaryMasks


class PostprocessorManifest(BaseModel):
    """Serialized postprocessor identity and configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    version: str
    params: dict[str, JsonValue] = Field(default_factory=dict)


class Postprocessor(ABC):
    """Convert continuous anomaly maps into binary segmentation masks."""

    KIND: ClassVar[str]
    VERSION: ClassVar[str]

    @abstractmethod
    def __call__(self, maps: AnomalyMaps) -> BinaryMasks: ...

    @abstractmethod
    def get_params(self) -> dict[str, JsonValue]: ...

    def _save(self, path: Path) -> None:
        """Save additional state, if any."""
        return

    def save(self, path: str | Path) -> PostprocessorManifest:
        """Save postprocessor state and return its manifest."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._save(path)

        return PostprocessorManifest(kind=self.KIND, version=self.VERSION, params=self.get_params())

    def _load(self, path: Path) -> None:
        """Restore postprocessor additional state, if any"""
        return

    @classmethod
    def load(cls, path: str | Path, manifest: PostprocessorManifest) -> Self:
        """Load a postprocessor from its artifact."""
        path = Path(path)

        if manifest.kind != cls.KIND:
            raise ValueError(
                f"Postprocessor kind mismatch: expected '{cls.KIND}', got '{manifest.kind}'"
            )

        postprocessor = cls(**manifest.params)
        postprocessor._load(path)

        return postprocessor
