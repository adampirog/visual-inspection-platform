import torch
from torch import nn


def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class ConvolutionalAutoencoder(nn.Module):
    """Reconstruct RGB images through a compressed spatial representation.

    Inputs have shape [N, 3, H, W] and values in [0, 1].
    Spatial dimensions must be divisible by 16.
    """

    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            _encoder_block(3, 32),
            _encoder_block(32, 64),
            _encoder_block(64, 128),
            _encoder_block(128, latent_channels),
        )

        self.decoder = nn.Sequential(
            _decoder_block(latent_channels, 128),
            _decoder_block(128, 64),
            _decoder_block(64, 32),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))
