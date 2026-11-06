from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator


# Convention - [batch_size, channels, height, width]


def _image(value: NDArray[np.uint32]) -> NDArray[np.uint32]:
    if value.dtype != np.dtype(np.uint32):
        raise ValueError("array dtype must be uint32")

    if value.ndim != 3:
        raise ValueError("array must be three-dimensional")

    return value


def _image_batch(value: NDArray[np.uint32]) -> NDArray[np.uint32]:
    if value.dtype != np.dtype(np.uint32):
        raise ValueError("array dtype must be uint32")

    if value.ndim != 3:
        raise ValueError("array must be four-dimensional")

    return value


def _float32_vector(value: NDArray[np.float32]) -> NDArray[np.float32]:
    if value.dtype != np.dtype(np.float32):
        raise ValueError("array dtype must be float32")

    if value.ndim != 1:
        raise ValueError("array must be one-dimensional")

    return value


NumpyImage = Annotated[NDArray[np.uint32], AfterValidator(_image)]
NumpyImageBatch = Annotated[NDArray[np.uint32], AfterValidator(_image_batch)]

Float32Vector = Annotated[NDArray[np.float32], AfterValidator(_float32_vector)]
