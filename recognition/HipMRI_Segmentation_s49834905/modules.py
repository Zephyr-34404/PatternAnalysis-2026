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

class CAN(nn.Module):
    """
    2D Context Aggregation Network for multi-class segmentation.

    A stack of 3x3 convolutions with exponentially increasing dilation
    (1, 2, 4, 8, 16, 32, 1). The receptive field grows to about 133 pixels
    while the feature maps stay at full input resolution, so there is no
    pooling, no upsampling and no skip connections. Each layer is
    convolution -> batch normalisation -> LeakyReLU.

    Reference:
        F. Yu and V. Koltun, "Multi-scale Context Aggregation by Dilated
        Convolutions," ICLR, 2016.

    Args:
        in_channels (int): input channels (1 for MRI).
        num_classes (int): output classes (6 for HipMRI).
        width (int): feature channels in every layer.
        dilations (tuple[int]): dilation rate of each context layer.

    Input:  (B, in_channels, H, W)
    Output: (B, num_classes, H, W) raw scores (logits), no softmax
    """

    def __init__(self, in_channels=1, num_classes=6, width=64,
                 dilations=(1, 2, 4, 8, 16, 32, 1)):
        super().__init__()

        def conv_block(c_in, c_out, dilation):
            # padding = dilation keeps height and width unchanged for a 3x3 kernel
            return [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers = conv_block(in_channels, width, 1)        # lift 1 channel to `width` features
        for d in dilations:                                # context aggregation
            layers += conv_block(width, width, d)
        layers += conv_block(width, width, 1)             # final 3x3 refinement
        layers.append(nn.Conv2d(width, num_classes, kernel_size=1))  # per-pixel class scores
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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

    can = CAN(in_channels=1, num_classes=6)
    y = can(x)
    print(f"\nCAN output: {tuple(y.shape)}  parameters: {count_parameters(can):,}")
    assert y.shape == (2, 6, 256, 128), "CAN output shape is wrong"
    print("OK")