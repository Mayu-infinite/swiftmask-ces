import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution:
    1. Depthwise spatial convolution.
    2. Pointwise 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DecoderBlock(nn.Module):
    """
    Upsamples decoder features, concatenates encoder skip
    features, and refines the combined representation.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.refine = DepthwiseSeparableConv(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)
        return self.refine(x)