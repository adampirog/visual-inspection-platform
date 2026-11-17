"""
Shared image preprocessing for training and inference.

Image sizes are expressed as (height, width).
Images use RGB channel-first float32 arrays in [0, 1].
Masks use boolean arrays, where True indicates an anomaly.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from visual_inspection_platform.types import BinaryMask, BinaryMasks, ImageBatch, ModelImage


def load_image(path: str | Path | BinaryIO, size: tuple[int, int]) -> ModelImage:
    """Decode and resize an image using bilinear interpolation."""
    height, width = size

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)

    return array.transpose(2, 0, 1) / 255.0


def load_mask(path: str | Path | BinaryIO, size: tuple[int, int]) -> BinaryMask:
    """
    Resize a grayscale mask using nearest-neighbor interpolation.

    Nonzero pixels represent anomalies.
    """
    height, width = size

    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(image) != 0


def load_images(paths: Sequence[str | Path | BinaryIO], size: tuple[int, int]) -> ImageBatch:
    height, width = size

    # preallocate continuous memory
    images = np.empty((len(paths), 3, height, width), dtype=np.float32)

    for index, path in enumerate(paths):
        images[index] = load_image(path, size)

    return images


def load_masks(paths: Sequence[str | Path | BinaryIO], size: tuple[int, int]) -> BinaryMasks:
    height, width = size

    # preallocate continuous memory
    masks = np.empty((len(paths), height, width), dtype=np.bool_)

    for index, path in enumerate(paths):
        masks[index] = load_mask(path, size)

    return masks
