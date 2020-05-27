"""Dense block"""
from torch import nn, Tensor, cat
from torch.nn import functional as f

from .bnrelu import BNRelu


class DenseBlock(nn.Module):
    """Dense unit for HoverNet"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1, bias=False)
        self.bn_relu1 = BNRelu(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5, padding=2, bias=False)
        self.bn_relu2 = BNRelu(32)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward"""
        out = self.bn_relu1(self.conv1(inputs))
        out = self.bn_relu2(self.conv2(out))

        return cat([out, f.interpolate(inputs, size=out.shape[2:])], dim=1)
