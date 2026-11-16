from .base import Postprocessor
from .spatial_threshold import SpatialThresholdPostprocessor
from .threshold import ThresholdPostprocessor


POSTPROCESSORS: dict[str, type[Postprocessor]] = {
    ThresholdPostprocessor.KIND: ThresholdPostprocessor,
    SpatialThresholdPostprocessor.KIND: SpatialThresholdPostprocessor,
}


def get_postprocessor_class(kind: str) -> type[Postprocessor]:
    try:
        return POSTPROCESSORS[kind]
    except KeyError as e:
        raise ValueError(f"Postprocessor of kind '{kind}' is not registered") from e
