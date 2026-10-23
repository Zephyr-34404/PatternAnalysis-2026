"""
modules.py — Model components for HipMRI 2D prostate segmentation.

Baseline: 2D U-Net (Ronneberger et al., 2015), adapted for 1-channel
256 x 128 MRI slices and 6 output classes. Convolutions use padding=1 so
feature maps keep their size, unlike the unpadded convolutions and
cropping in the original paper.

The model uses only PyTorch (no NumPy), as required by the specification.

Reference:
    O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for
    Biomedical Image Segmentation," MICCAI, 2015.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two 3x3 convolutions, each followed by batch normalisation and ReLU.

    Spatial size is unchanged; channels go from in_channels to out_channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    Encoder step: 2x2 max-pool (halves height and width), then DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: 2x2 transposed convolution (doubles height and width,
    halves channels), concatenate the matching encoder feature map
    (skip connection), then DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    2D U-Net for multi-class segmentation.

    Args:
        in_channels (int): input channels (1 for MRI).
        num_classes (int): output classes (6 for HipMRI).
        base_channels (int): channels after the first block; doubles at
            each of the four encoder levels.

    Input:  (B, in_channels, H, W), H and W divisible by 16
    Output: (B, num_classes, H, W) raw scores (logits), no softmax
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=32):
        super().__init__()
        c = base_channels
        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, 2 * c)
        self.down2 = Down(2 * c, 4 * c)
        self.down3 = Down(4 * c, 8 * c)
        self.down4 = Down(8 * c, 16 * c)

        # Decoder
        self.up1 = Up(16 * c, 8 * c)
        self.up2 = Up(8 * c, 4 * c)
        self.up3 = Up(4 * c, 2 * c)
        self.up4 = Up(2 * c, c)

        # Per-pixel class scores
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder: keep each output for the skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder: upsample and merge with the matching encoder output
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test: random input, check output shape.
    model = UNet(in_channels=1, num_classes=6, base_channels=32)
    x = torch.randn(2, 1, 256, 128)
    y = model(x)
    print(f"input : {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}  (expected (2, 6, 256, 128))")
    print(f"trainable parameters: {count_parameters(model):,}")
    assert y.shape == (2, 6, 256, 128), "output shape is wrong"
    print("OK")