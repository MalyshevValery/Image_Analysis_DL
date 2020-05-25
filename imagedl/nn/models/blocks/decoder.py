"""Decoder for HoverNet"""
from typing import Tuple

from torch import nn, Tensor
from torch.nn import functional as f

from .dense import DenseBlock


class Decoder(nn.Module):
    """HoverNet Decoder"""

    def __init__(self, in_channels: int, n_dense_1: int = 2,
                 n_dense_2: int = 2):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=5, padding=2)
        self.dense_block1 = nn.Sequential(
            DenseBlock(256),
            *[DenseBlock(256 + 32 * i) for i in range(1, n_dense_1)]
        )
        self.conv2 = nn.Conv2d(256 + 32 * n_dense_1, 512, 1)
        self.conv3 = nn.Conv2d(512, 128, 5, padding=2)
        self.dense_block2 = nn.Sequential(
            DenseBlock(128),
            *[DenseBlock(128 + 32 * i) for i in range(1, n_dense_2)]
        )
        self.conv4 = nn.Conv2d(128 + 32 * n_dense_2, 256, kernel_size=1)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=5, padding=2)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        """Forward"""
        d1, d2, d3, d4 = inputs
        u3 = f.interpolate(d4, scale_factor=2) + d3
        u3 = self.conv1(u3)
        u3 = self.dense_block1(u3)
        u3 = self.conv2(u3)

        u2 = f.interpolate(u3, scale_factor=2) + d2
        u2x = self.conv3(u2)
        u2 = self.dense_block2(u2x)
        u2 = self.conv4(u2)

        u1 = f.interpolate(u2, scale_factor=2) + d1
        u1 = self.conv5(u1)
        return u1
