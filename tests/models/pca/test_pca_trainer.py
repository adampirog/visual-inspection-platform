import numpy as np
import pytest
from PIL import Image

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.models.pca.trainer import PCAConfig, PCATrainer


@pytest.fixture
def datasets(tmp_path):
    rows = ["object,split,label,image,mask"]

    for i, value in enumerate((0, 80, 160)):
        image = np.full((4, 4, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(tmp_path / f"train_{i}.png")
        rows.append(f"cashew,train,normal,train_{i}.png,")

    normal = np.zeros((4, 4, 3), dtype=np.uint8)
    anomaly = normal.copy()
    anomaly[1, 2] = 255

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 2] = 255

    Image.fromarray(normal).save(tmp_path / "normal.png")
    Image.fromarray(anomaly).save(tmp_path / "anomaly.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")

    rows.extend(
        [
            "cashew,test,normal,normal.png,",
            "cashew,test,anomaly,anomaly.png,mask.png",
        ]
    )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return (
        ImageRecordsDataset(manifest, tmp_path, split="train"),
        ImageRecordsDataset(manifest, tmp_path, split="test"),
    )


@pytest.fixture
def trainer():
    return PCATrainer(PCAConfig(n_components=1, image_size=(4, 4)))


def test_fit(trainer, datasets):
    train, _ = datasets
    model = trainer.fit(train)

    assert model.components.shape == (1, 48)
    assert model.mean.shape == (48,)
    assert model.training_metadata.config["n_components"] == 1

    maps = model.predict(train.load_images((4, 4)))
    np.testing.assert_allclose(maps, 0.0, atol=1e-6)


def test_evaluate(trainer, datasets):
    train, test = datasets
    model = trainer.fit(train)

    result = trainer.evaluate(model, test, batch_size=1)

    assert "binary" not in result
    assert result["continuous"]["cashew"] == pytest.approx({"pixel_auroc": 1.0, "pixel_ap": 1.0})
