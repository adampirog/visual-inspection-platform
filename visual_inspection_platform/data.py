import csv
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, override


@dataclass(frozen=True, slots=True)
class ImageData:
    path: Path
    mask: Path | None
    label: str


class ImageDataset(Sequence[ImageData]):
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

        objects = set(objects) if objects else None
        self.images = self.read_manifest(manifest, split, objects)

    def read_manifest(
        self, manifest: str | Path, split: str, objects: set[str] | None
    ) -> list[ImageData]:
        result = []
        with open(manifest, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["split"] != split:
                    continue
                if objects is not None and row["object"] not in objects:
                    continue

                if row["mask"]:
                    mask = self.data_dir / row["mask"]
                else:
                    mask = None

                result.append(
                    ImageData(path=self.data_dir / row["image"], mask=mask, label=row["label"])
                )

        return result

    @override
    def __len__(self) -> int:
        return len(self.images)

    @overload
    def __getitem__(self, key: int) -> ImageData:
        pass

    @overload
    def __getitem__(self, key: slice) -> Sequence[ImageData]:
        pass

    @override
    def __getitem__(self, key: int | slice) -> ImageData | Sequence[ImageData]:
        return self.images[key]
