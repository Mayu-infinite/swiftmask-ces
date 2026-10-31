import torch
import torch.nn as nn
import torch.nn.functional as F

from ces.methods.track_b.blocks import (
    DepthwiseSeparableConv,
    DecoderBlock,
)


class SwiftMaskFCN(nn.Module):
    """
    Original SwiftMask FCN (V1).

    Lightweight encoder-decoder segmentation network
    using depthwise-separable convolutions and skip connections.
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = DepthwiseSeparableConv(3, 16)
        self.encoder2 = DepthwiseSeparableConv(16, 32)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 64),
        )

        # Decoder
        self.decoder2 = DecoderBlock(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
        )

        self.decoder1 = DecoderBlock(
            in_channels=32,
            skip_channels=16,
            out_channels=16,
        )

        # Binary segmentation output
        self.output_projection = nn.Conv2d(
            16,
            1,
            kernel_size=1,
        )

    def forward(self, x):

        # Encoder
        encoder1_features = self.encoder1(x)

        pooled1 = F.max_pool2d(
            encoder1_features,
            kernel_size=2,
            stride=2,
        )

        encoder2_features = self.encoder2(pooled1)

        pooled2 = F.max_pool2d(
            encoder2_features,
            kernel_size=2,
            stride=2,
        )

        # Bottleneck
        bottleneck_features = self.bottleneck(pooled2)

        # Decoder with skip connections
        decoder2_features = self.decoder2(
            bottleneck_features,
            encoder2_features,
        )

        decoder1_features = self.decoder1(
            decoder2_features,
            encoder1_features,
        )

        # Output logits
        return self.output_projection(decoder1_features)