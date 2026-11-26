import torch
from torch import nn

from .network import ConvolutionalAutoencoder


class _AnomalyMapModule(nn.Module):
    def __init__(self, network: ConvolutionalAutoencoder) -> None:
        super().__init__()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.network(images)).square().mean(dim=1)


def export_onnx(network: ConvolutionalAutoencoder, image_size: tuple[int, int]) -> bytes:
    """Export the autoencoder as a self-contained ONNX anomaly detector.

    The network must be on CPU, in FP32 and evaluation mode.
    The exported graph supports dynamic batch sizes and fixed spatial dimensions.
    """
    module = _AnomalyMapModule(network).eval()
    example = torch.rand(2, 3, *image_size, dtype=torch.float32)

    program = torch.onnx.export(
        module,
        (example,),
        dynamo=True,
        input_names=["images"],
        output_names=["anomaly_maps"],
        dynamic_shapes={
            "images": {0: torch.export.Dim("batch_size", min=1)},
        },
        optimize=True,
        external_data=False,
    )

    assert program is not None

    return program.model_proto.SerializeToString()
