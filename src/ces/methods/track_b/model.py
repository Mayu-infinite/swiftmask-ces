
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 convolution followed by pointwise channel mixing."""

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        padding = dilation

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)

        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.activation(x)
        return x


class RSUBlock(nn.Module):
    """
    Lightweight Residual U-block.

    Each block has a small encoder-decoder inside it. The outer residual
    connection preserves the input representation while the inner U-path
    captures features at multiple spatial scales.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.enc1 = DepthwiseSeparableConv(out_channels, mid_channels)
        self.enc2 = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.bottleneck = DepthwiseSeparableConv(
            mid_channels, mid_channels, dilation=2
        )

        self.dec2 = DepthwiseSeparableConv(
            mid_channels * 2, mid_channels
        )
        self.dec1 = DepthwiseSeparableConv(
            mid_channels * 2, out_channels
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    @staticmethod
    def _upsample_to(x, reference):
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        residual = self.input_projection(x)

        enc1 = self.enc1(residual)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))

        dec2_input = torch.cat(
            [self._upsample_to(bottleneck, enc2), enc2], dim=1
        )
        dec2 = self.dec2(dec2_input)

        dec1_input = torch.cat(
            [self._upsample_to(dec2, enc1), enc1], dim=1
        )
        dec1 = self.dec1(dec1_input)

        return dec1 + residual


class SwiftMaskFCN(nn.Module):
    """
    SwiftMask V2: compact U²-Net-inspired salient-object segmentation model.

    The class name is retained for compatibility with the existing
    training/evaluation imports. The architecture is newly initialized.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder: progressively wider features at lower resolutions.
        self.stage1 = RSUBlock(in_channels, 8, 16)
        self.stage2 = RSUBlock(16, 12, 24)
        self.stage3 = RSUBlock(24, 16, 32)
        self.stage4 = RSUBlock(32, 24, 48)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.bottleneck = RSUBlock(48, 32, 64)

        # Decoder: concatenate the matching encoder feature at each scale.
        self.decode4 = RSUBlock(64 + 48, 24, 48)
        self.decode3 = RSUBlock(48 + 32, 16, 32)
        self.decode2 = RSUBlock(32 + 24, 12, 24)
        self.decode1 = RSUBlock(24 + 16, 8, 16)

        self.output_head = nn.Conv2d(
            16, out_channels, kernel_size=1
        )

    @staticmethod
    def _upsample_to(x, reference):
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        original_size = x.shape[-2:]

        enc1 = self.stage1(x)
        enc2 = self.stage2(self.pool(enc1))
        enc3 = self.stage3(self.pool(enc2))
        enc4 = self.stage4(self.pool(enc3))

        center = self.bottleneck(self.pool(enc4))

        dec4 = self.decode4(
            torch.cat([self._upsample_to(center, enc4), enc4], dim=1)
        )
        dec3 = self.decode3(
            torch.cat([self._upsample_to(dec4, enc3), enc3], dim=1)
        )
        dec2 = self.decode2(
            torch.cat([self._upsample_to(dec3, enc2), enc2], dim=1)
        )
        dec1 = self.decode1(
            torch.cat([self._upsample_to(dec2, enc1), enc1], dim=1)
        )

        logits = self.output_head(dec1)

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


# Compatibility alias for code that prefers the architecture-specific name.
SwiftMaskU2Net = SwiftMaskFCN
