import torch
from torch.utils.data import Dataset, Subset, random_split

from visual_inspection_platform.data import ImageRecordsDataset


class ImageTensorDataset(Dataset[torch.Tensor]):
    def __init__(self, data: ImageRecordsDataset, image_size: tuple[int, int]) -> None:
        self.data = data
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.data[index].load_image(self.image_size))

    def split(
        self,
        validation_fraction: float,
        *,
        random_state: int | None = None,
    ) -> tuple[Subset[torch.Tensor], Subset[torch.Tensor]]:
        """Randomly split the dataset into training and validation subsets.

        Both subsets contain at least one sample and share the underlying
        dataset without loading or copying images.
        """
        if len(self) < 2:
            raise ValueError("At least two samples are required")

        if not 0 < validation_fraction < 1:
            raise ValueError("Validation fraction must be between 0 and 1")

        n_val = max(1, min(len(self) - 1, round(len(self) * validation_fraction)))

        rng = torch.Generator().manual_seed(random_state) if random_state is not None else None
        train, val = random_split(self, [len(self) - n_val, n_val], generator=rng)

        return train, val
