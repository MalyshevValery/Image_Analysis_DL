"""Encoder for HoverNet"""
from typing import Tuple

from torch import Tensor, nn

from .residual import ResidualBlock


class Encoder(nn.Module):
    """HoverNet encoder"""

    def __init__(self, increased: bool = False, bias: bool = False) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=bias)
        self.residual_block1 = nn.Sequential(
            ResidualBlock(64, 256, bias=bias),
            ResidualBlock(256, 256, bias=bias),
            ResidualBlock(256, 256, bias=bias) if increased else nn.Identity(),
        )
        self.residual_block2 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, bias=bias),
            ResidualBlock(512, 512, bias=bias),
            ResidualBlock(512, 512, bias=bias) if increased else nn.Identity(),
            ResidualBlock(512, 512, bias=bias) if increased else nn.Identity(),
        )
        self.residual_block3 = nn.Sequential(
            ResidualBlock(512, 1024, stride=2, bias=bias),
            ResidualBlock(1024, 1024, bias=bias),
            ResidualBlock(1024, 1024, bias=bias),
            ResidualBlock(1024, 1024,
                          bias=bias) if increased else nn.Identity(),
            ResidualBlock(1024, 1024,
                          bias=bias) if increased else nn.Identity(),
            ResidualBlock(1024, 1024,
                          bias=bias) if increased else nn.Identity(),
        )
        self.residual_block4 = nn.Sequential(
            ResidualBlock(1024, 2048, stride=2, bias=bias),
            ResidualBlock(2048, 2048, bias=bias),
            ResidualBlock(2048, 2048,
                          bias=bias) if increased else nn.Identity(),
        )
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, bias=bias)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        """Forward"""
        d1 = self.conv1(inputs)
        d1 = self.residual_block1(d1)
        d2 = self.residual_block2(d1)
        d3 = self.residual_block3(d2)
        d4 = self.residual_block4(d3)
        d4 = self.conv2(d4)
        return d1, d2, d3, d4
