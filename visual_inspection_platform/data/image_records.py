import csv
from collections.abc import Collection, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Self, overload, override

import numpy as np

from visual_inspection_platform.types import BinaryMask, BinaryMasks, ImageBatch, ModelImage
from visual_inspection_platform.utils.images import load_image, load_mask


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path
    mask: Path | None
    is_anomaly: bool
    category: str

    def load_image(self, size: tuple[int, int]) -> ModelImage:
        """Load an RGB CHW float32 image in [0, 1] at size (height, width)."""
        return load_image(self.path, size=size)

    def load_mask(self, size: tuple[int, int]) -> BinaryMask:
        """Load the mask at size (height, width).

        Missing masks yield all-False masks for normal images and raise
        ValueError for anomalous images.
        """
        height, width = size

        if self.mask is None:
            if self.is_anomaly:
                raise ValueError(f"Missing anomaly mask for {self.path}")

            return np.zeros((height, width), dtype=np.bool_)

        return load_mask(self.mask, size=size)


class ImageRecordsDataset(Sequence[ImageRecord]):
    """Sequence of image records selected from a CSV manifest.

    Expected columns: object, split, label, image, mask. Paths are relative
    to data_dir; an empty mask field indicates a missing mask. label_map maps
    labels to booleans, where True indicates an anomaly. objects=None selects
    all categories; an empty collection selects none. Images and masks are
    loaded on demand, with sizes given as (height, width).
    """

    def __init__(
        self,
        manifest: str | Path,
        data_dir: str | Path,
        label_map: dict[str, bool],
        *,
        split: str = "train",
        objects: Collection[str] | None = None,
    ) -> None:

        if isinstance(objects, str):
            objects = {objects}
        elif objects is not None:
            objects = set(objects)

        self.split = split
        self.objects = objects
        self.label_map = label_map
        self.data_dir = Path(data_dir)
        self.images = tuple(self.read_manifest(manifest))

    def read_manifest(self, manifest: str | Path) -> list[ImageRecord]:
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
                        is_anomaly=self.label_map[row["label"]],
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
        """Load all selected images into memory in dataset order."""
        height, width = size

        # preallocate memory
        images = np.empty((len(self), 3, height, width), dtype=np.float32)

        for i, record in enumerate(self):
            images[i] = record.load_image(size)

        return images

    def load_masks(self, size: tuple[int, int]) -> BinaryMasks:
        """Load all selected masks into memory in dataset order."""
        height, width = size

        # preallocate memory
        masks = np.empty((len(self), height, width), dtype=np.bool_)

        for i, record in enumerate(self):
            masks[i] = record.load_mask(size)

        return masks
