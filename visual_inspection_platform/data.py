import csv
from collections.abc import Collection, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Self, overload, override

import numpy as np
from PIL import Image

from visual_inspection_platform.types import BinaryMask, BinaryMasks, ImageBatch, ModelImage


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Path and annotations for a single image."""

    path: Path
    mask: Path | None
    label: str
    category: str

    def load_image(self, size: tuple[int, int] = (64, 64)) -> ModelImage:
        """Load an RGB image as CHW float32 in [0, 1]."""
        height, width = size

        with Image.open(self.path) as image:
            image = image.convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)

        return array.transpose(2, 0, 1) / 255.0

    def load_mask(self, size: tuple[int, int] = (64, 64)) -> BinaryMask:
        """Load a binary mask; missing masks represent normal images."""
        height, width = size

        if self.mask is None:
            return np.zeros((height, width), dtype=np.bool_)

        with Image.open(self.mask) as image:
            image = image.convert("L")
            image = image.resize((width, height), Image.Resampling.NEAREST)
            return np.asarray(image) != 0


class ImageRecordsDataset(Sequence[ImageRecord]):
    """Sequence of image records loaded from a CSV manifest."""

    def __init__(
        self,
        manifest: str | Path,
        data_dir: str | Path,
        *,
        split: str = "train",
        objects: Collection[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        if not objects:
            objects = None
        elif isinstance(objects, str):
            objects = {objects}
        else:
            objects = set(objects)

        self.objects = objects
        self.images = tuple(self.read_manifest(manifest))

    def read_manifest(self, manifest: str | Path) -> list[ImageRecord]:
        """Read records matching the selected split and categories."""
        result = []
        with open(manifest, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["split"] != self.split:
                    continue
                if self.objects is not None and row["object"] not in self.objects:
                    continue

                if row["mask"]:
                    mask = self.data_dir / row["mask"]
                else:
                    mask = None

                result.append(
                    ImageRecord(
                        path=self.data_dir / row["image"],
                        mask=mask,
                        label=row["label"],
                        category=row["object"],
                    )
                )

        return result

    @override
    def __len__(self) -> int:
        return len(self.images)

    @overload
    def __getitem__(self, key: int) -> ImageRecord: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @override
    def __getitem__(self, key: int | slice) -> ImageRecord | Self:
        if isinstance(key, slice):
            subset = copy(self)
            subset.images = self.images[key]
            return subset

        return self.images[key]

    def load_images(self, size: tuple[int, int]) -> ImageBatch:
        height, width = size

        # preallocate memory
        images = np.empty((len(self), 3, height, width), dtype=np.float32)

        for i, record in enumerate(self):
            images[i] = record.load_image(size)

        return images

    def load_masks(self, size: tuple[int, int]) -> BinaryMasks:
        height, width = size

        # preallocate memory
        masks = np.empty((len(self), height, width), dtype=np.bool_)

        for i, record in enumerate(self):
            masks[i] = record.load_mask(size)

        return masks
