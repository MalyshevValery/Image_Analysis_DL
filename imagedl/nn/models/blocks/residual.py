"""Residual Unit for HoverNet"""
from torch import nn, Tensor

from .bnrelu import BNRelu


class ResidualBlock(nn.Module):
    """Residual Unit"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expansion: int = 4, bias: bool = False):
        super().__init__()
        bottleneck_channels = out_channels // expansion

        self.bn_relu1 = BNRelu(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=bias)
        self.bn_relu2 = BNRelu(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3,
                               stride=stride, padding=1, bias=bias)

        self.bn_relu3 = BNRelu(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=bias)

        if in_channels != out_channels or stride != 1:
            self.shortcut: nn.Module = nn.Conv2d(in_channels, out_channels, 1,
                                                 stride=stride, bias=bias)
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward"""
        out: Tensor = self.bn_relu1(inputs)
        shortcut = self.shortcut(inputs)
        out = self.bn_relu2(self.conv1(out))
        out = self.bn_relu3(self.conv2(out))
        out = self.conv3(out)
        out += shortcut
        return out
