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
    """Convert anomaly maps into boolean segmentation masks.

    Input and output shapes are identical: [N, H, W].
    True indicates an anomalous pixel.
    """

    KIND: ClassVar[str]
    VERSION: ClassVar[str]

    @abstractmethod
    def __call__(self, maps: AnomalyMaps) -> BinaryMasks: ...

    @abstractmethod
    def get_params(self) -> dict[str, JsonValue]:
        """Return JSON-compatible constructor arguments required for deserialization."""

    def _save(self, path: Path) -> None:
        """Save state that cannot be represented by constructor parameters."""
        return

    def save(self, path: str | Path) -> PostprocessorManifest:
        """Save additional state and return the manifest for embedding in a model artifact."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._save(path)

        return PostprocessorManifest(kind=self.KIND, version=self.VERSION, params=self.get_params())

    def _load(self, path: Path) -> None:
        """Restore additional state after construction from manifest parameters."""
        return

    @classmethod
    def load(cls, path: str | Path, manifest: PostprocessorManifest) -> Self:
        """Restore a postprocessor from its manifest and optional state directory.

        The manifest kind must match the class.
        """
        path = Path(path)

        if manifest.kind != cls.KIND:
            raise ValueError(
                f"Postprocessor kind mismatch: expected '{cls.KIND}', got '{manifest.kind}'"
            )

        if manifest.version != cls.VERSION:
            raise ValueError(
                f"Postprocessor version mismatch: expected '{cls.VERSION}', "
                f"got '{manifest.version}'"
            )

        postprocessor = cls(**manifest.params)
        postprocessor._load(path)

        return postprocessor
