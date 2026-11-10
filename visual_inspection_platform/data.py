import csv
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, override


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Path and annotations for a single image."""

    path: Path
    mask: Path | None
    label: str
    category: str


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
    def __getitem__(self, key: int) -> ImageRecord:
        pass

    @overload
    def __getitem__(self, key: slice) -> Sequence[ImageRecord]:
        pass

    @override
    def __getitem__(self, key: int | slice) -> ImageRecord | Sequence[ImageRecord]:
        return self.images[key]
