"""Segmentation Head"""
from torch import Tensor, nn

from .bnrelu import BNRelu


class SegmentationHead(nn.Module):
    """Last part of NN to perform segmentation"""

    def __init__(self, n_channels: int = 1):
        super(SegmentationHead, self).__init__()
        self.bn_relu = BNRelu(num_features=64)
        self.conv = nn.Conv2d(64, n_channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward"""
        out: Tensor = self.bn_relu(inputs)
        out = self.conv(out)
        return out
