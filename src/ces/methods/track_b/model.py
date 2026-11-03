import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 convolution followed by a pointwise mixing layer."""

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


class MultiScaleContext(nn.Module):
    """Dilated multi-scale bottleneck used to widen receptive field cheaply."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = DepthwiseSeparableConv(in_channels, 16, dilation=1)
        self.branch2 = DepthwiseSeparableConv(in_channels, 16, dilation=2)
        self.branch3 = DepthwiseSeparableConv(in_channels, 16, dilation=4)

        self.project = nn.Sequential(
            nn.Conv2d(48, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        combined = torch.cat([x1, x2, x3], dim=1)
        projected = self.project(combined)
        residual = self.residual(x)

        return projected + residual


class RSUBlock(nn.Module):
    """Compact residual U-block with a lightweight encoder-decoder interface."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.enc1 = DepthwiseSeparableConv(out_channels, mid_channels)
        self.enc2 = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.dec1 = DepthwiseSeparableConv(mid_channels + out_channels, out_channels)

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
        up = self._upsample_to(enc2, enc1)
        merged = torch.cat([up, enc1], dim=1)
        decoded = self.dec1(merged)
        return decoded + residual


class SwiftMaskFCN(nn.Module):
    """Compact residual U-Net with multi-scale bottleneck inspired by the assignment reference."""

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.stage1 = RSUBlock(in_channels, 16, 16)
        self.stage2 = RSUBlock(16, 24, 24)
        self.stage3 = RSUBlock(24, 40, 40)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.bottleneck = MultiScaleContext(40, 56)

        self.decode3 = RSUBlock(56 + 40, 40, 40)
        self.decode2 = RSUBlock(40 + 24, 24, 24)
        self.decode1 = RSUBlock(24 + 16, 16, 16)
        self.output_head = nn.Conv2d(16, out_channels, kernel_size=1)

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

        center = self.bottleneck(self.pool(enc3))

        dec3 = self.decode3(torch.cat([self._upsample_to(center, enc3), enc3], dim=1))
        dec2 = self.decode2(torch.cat([self._upsample_to(dec3, enc2), enc2], dim=1))
        dec1 = self.decode1(torch.cat([self._upsample_to(dec2, enc1), enc1], dim=1))

        logits = self.output_head(dec1)

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


SwiftMaskU2Net = SwiftMaskFCN
