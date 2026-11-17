from collections.abc import Collection
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from visual_inspection_platform.types import BinaryMask, BinaryMasks, ImageBatch, ModelImage


def load_image(path: str | Path | BinaryIO, size: tuple[int, int] = (64, 64)) -> ModelImage:
    """Load an RGB image as CHW float32 in [0, 1]."""
    height, width = size

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)

    return array.transpose(2, 0, 1) / 255.0


def load_mask(path: str | Path | BinaryIO, size: tuple[int, int] = (64, 64)) -> BinaryMask:
    """Load a binary mask; missing masks represent normal images."""
    height, width = size

    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(image) != 0


def load_images(paths: Collection[str | Path | BinaryIO], size: tuple[int, int]) -> ImageBatch:
    height, width = size

    # preallocate continuous memory
    images = np.empty((len(paths), 3, height, width), dtype=np.float32)

    for index, path in enumerate(paths):
        images[index] = load_image(path, size)

    return images


def load_masks(paths: Collection[str | Path | BinaryIO], size: tuple[int, int]) -> BinaryMasks:
    height, width = size

    # preallocate continuous memory
    masks = np.empty((len(paths), height, width), dtype=np.bool_)

    for index, path in enumerate(paths):
        masks[index] = load_mask(path, size)

    return masks
