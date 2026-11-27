import torch


def test_reconstruction(network, image_batch, cae_config):
    images = torch.from_numpy(image_batch[:2])

    with torch.inference_mode():
        latent = network.encode(images)
        reconstruction = network(images)

    height, width = cae_config.image_size

    assert latent.shape == (2, cae_config.latent_channels, height // 16, width // 16)

    assert reconstruction.shape == images.shape
    assert reconstruction.dtype == torch.float32
    assert torch.isfinite(reconstruction).all()
    assert ((reconstruction >= 0) & (reconstruction <= 1)).all()


def test_anomaly_maps(network, image_batch):
    from visual_inspection_platform.models.cae.network import CAEAnomalyModule

    images = torch.from_numpy(image_batch[:2])
    module = CAEAnomalyModule(network).eval()

    with torch.inference_mode():
        reconstruction = network(images)
        maps = module(images)

    expected = (images - reconstruction).square().sum(dim=1) / 3

    assert maps.shape == (2, *images.shape[2:])
    torch.testing.assert_close(maps, expected)
