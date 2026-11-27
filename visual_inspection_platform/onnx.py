import torch
from torch import nn


def export_anomaly_detector(network: nn.Module, image_size: tuple[int, int]) -> bytes:
    """Export an anomaly detector as a self-contained ONNX graph.

    The network must be in FP32 evaluation mode on CPU, accepting
    [N, 3, H, W] images and returning [N, H, W] anomaly maps.
    The batch dimension is dynamic; spatial dimensions are fixed.
    """
    example = torch.zeros(2, 3, *image_size, dtype=torch.float32)

    program = torch.onnx.export(
        network,
        (example,),
        dynamo=True,
        input_names=["images"],
        output_names=["anomaly_maps"],
        dynamic_shapes=({0: torch.export.Dim("batch_size", min=1)},),
        optimize=True,
        external_data=False,
    )

    if program is None:
        raise RuntimeError("ONNX export failed")

    return program.model_proto.SerializeToString()
