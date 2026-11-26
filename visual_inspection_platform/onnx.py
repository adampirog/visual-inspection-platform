import torch
from torch import nn


def export_onnx(network: nn.Module, image_size: tuple[int, int]) -> bytes:
    example = torch.rand(2, 3, *image_size, dtype=torch.float32)

    program = torch.onnx.export(
        network.eval(),
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

    if program is None:
        raise RuntimeError("ONNX export failed")

    return program.model_proto.SerializeToString()
