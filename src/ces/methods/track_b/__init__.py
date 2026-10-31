
from ces.methods.track_b.model import (
    SwiftMaskFCN,
    SwiftMaskU2Net,
    RSUBlock,
    DepthwiseSeparableConv,
)


def count_parameters(model):
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


from ces.methods.track_b.losses import (
    SoftDiceLoss,
    SegmentationLoss,
)


__all__ = [
    "SwiftMaskFCN",
    "SwiftMaskU2Net",
    "RSUBlock",
    "DepthwiseSeparableConv",
    "count_parameters",
    "SoftDiceLoss",
    "SegmentationLoss",
]
