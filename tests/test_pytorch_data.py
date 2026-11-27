import pytest
import torch
from torch.utils.data import DataLoader

from visual_inspection_platform.data.pytorch_data import ImageTensorDataset


@pytest.fixture
def dataset(dataset_factory):
    return ImageTensorDataset(dataset_factory(split="train"), image_size=(32, 48))


def test_image_loading(dataset):
    images = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    expected = torch.from_numpy(dataset.data.load_images(dataset.image_size))

    assert images.dtype == torch.float32
    assert images.shape == (4, 3, 32, 48)

    torch.testing.assert_close(images, expected)


@pytest.mark.parametrize(
    "fraction, expected_size",
    [(0.01, 1), (0.5, 2), (0.99, 3)],
)
def test_split(dataset, fraction, expected_size):
    train, val = dataset.split(fraction, random_state=42)

    assert len(val) == expected_size
    assert len(train) + len(val) == len(dataset)

    assert train.dataset is dataset
    assert val.dataset is dataset

    assert set(train.indices).isdisjoint(val.indices)
    assert sorted(train.indices + val.indices) == list(range(len(dataset)))

    torch.testing.assert_close(train[0], dataset[train.indices[0]])


def test_reproducible_split(dataset):
    train_a, val_a = dataset.split(0.5, random_state=42)
    train_b, val_b = dataset.split(0.5, random_state=42)

    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices


def test_unseeded_split(dataset):
    train, val = dataset.split(0.5)

    assert sorted(train.indices + val.indices) == list(range(len(dataset)))


def test_split_single_sample(dataset_factory):
    dataset = ImageTensorDataset(dataset_factory(objects="pcb1"), image_size=(32, 48))

    with pytest.raises(ValueError, match="At least two samples"):
        dataset.split(0.2)
