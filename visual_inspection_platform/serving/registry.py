from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module
from threading import RLock

from visual_inspection_platform.models import AnomalyDetector


@dataclass(frozen=True, slots=True)
class ModelSpec:
    module: str
    class_name: str


class ModelRegistry(Mapping[str, type[AnomalyDetector]]):
    """Registry of detector classes keyed by model family.

    Implementations are imported and cached on first lookup, not on registration.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._model_types: dict[str, type[AnomalyDetector]] = {}

        self._lock = RLock()

    def register(self, model_family: str, *, module: str, class_name: str) -> None:
        """Register or replace a model family and invalidate its cached class."""
        spec = ModelSpec(module=module, class_name=class_name)

        with self._lock:
            self._specs[model_family] = spec
            self._model_types.pop(model_family, None)

    def __getitem__(self, model_family: str) -> type[AnomalyDetector]:
        with self._lock:
            if model_family in self._model_types:
                return self._model_types[model_family]

            spec = self._specs[model_family]
            module = import_module(spec.module)
            model_class = getattr(module, spec.class_name)

            self._model_types[model_family] = model_class

            return model_class

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            families = tuple(self._specs)

        return iter(families)

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)


def create_default_registry() -> ModelRegistry:
    """Create a registry containing all built-in detector families."""
    registry = ModelRegistry()
    registry.register(
        "pca",
        module="visual_inspection_platform.models.pca.model",
        class_name="PCADetector",
    )
    return registry
