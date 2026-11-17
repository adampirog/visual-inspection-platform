import numpy as np
from PIL import Image

from visual_inspection_platform.utils.images import load_image, load_images, load_masks


def test_image_preprocessing(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (6, 4), (255, 128, 0)).save(path)

    image = load_image(path, (2, 3))

    assert image.shape == (3, 2, 3)
    assert image.dtype == np.float32

    expected = np.broadcast_to(
        np.array([1.0, 128 / 255, 0.0])[:, None, None],
        image.shape,
    )
    np.testing.assert_allclose(image, expected)

    with path.open("rb") as stream:
        batch = load_images([path, stream], (2, 3))

    np.testing.assert_array_equal(batch, np.stack([image, image]))
    assert batch.flags.c_contiguous


def test_mask_preprocessing(dataset_factory):
    dataset = dataset_factory(split="test")
    path = dataset[1].mask

    masks = load_masks([path], (8, 12))

    expected = np.zeros((1, 8, 12), dtype=bool)
    expected[0, 2:4, 4:6] = True

    assert masks.dtype == np.bool_
    np.testing.assert_array_equal(masks, expected)
