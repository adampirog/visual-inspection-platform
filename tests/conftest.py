import numpy as np
import pytest
from PIL import Image

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.models.pca import PCADetector


@pytest.fixture
def dataset_factory(tmp_path):
    rows = ["object,split,label,image,mask"]

    for i, value in enumerate((0, 80, 160)):
        image = np.full((4, 6, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(tmp_path / f"train_{i}.png")
        rows.append(f"cashew,train,good,train_{i}.png,")

    normal = np.zeros((4, 6, 3), dtype=np.uint8)
    anomaly = normal.copy()
    anomaly[1, 2] = 255

    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1, 2] = 255

    Image.fromarray(normal).save(tmp_path / "normal.png")
    Image.fromarray(anomaly).save(tmp_path / "anomaly.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")

    rows.extend(
        [
            "cashew,test,good,normal.png,",
            "cashew,test,bad,anomaly.png,mask.png",
            "pcb1,train,good,normal.png,",
        ]
    )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def create_dataset(**kwargs):
        return ImageRecordsDataset(
            manifest,
            tmp_path,
            label_map={"good": False, "bad": True},
            **kwargs,
        )

    return create_dataset


@pytest.fixture
def make_pca():
    def create_model(image_size=(2, 3), **kwargs):
        height, width = image_size
        features = 3 * height * width

        return PCADetector(
            image_size=image_size,
            mean=np.zeros(features, dtype=np.float32),
            components=np.eye(1, features, dtype=np.float32),
            **kwargs,
        )

    return create_model
