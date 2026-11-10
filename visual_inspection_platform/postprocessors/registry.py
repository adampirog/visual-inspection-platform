from .base import Postprocessor


POSTPROCESSORS: dict[str, type[Postprocessor]] = {}


def get_postprocessor_class(kind: str) -> type[Postprocessor]:
    try:
        return POSTPROCESSORS[kind]
    except KeyError as e:
        raise ValueError(f"Postprocessor of kind '{kind}' is not registered") from e
