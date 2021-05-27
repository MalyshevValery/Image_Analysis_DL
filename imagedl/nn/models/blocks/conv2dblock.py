"""Parts for NN nn"""
from typing import Type

from torch import nn, Tensor


class Conv2dBlock(nn.Module):
    """
    Torch layer which includes some 2D convolution layers of same kernel and
    channel size, activation function and batch normalization parameter

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param layers: number of convolutions in block
    :param activation: Activations function
    :param kernel_size: Size of kernel
    :param batchnorm: True if batch normalization should be applied
    """

    def __init__(self, in_channels: int, out_channels: int, layers: int = 2,
                 activation: Type[nn.Module] = nn.ReLU, kernel_size: int = 3,
                 batchnorm: bool = True):
        super(Conv2dBlock, self).__init__()
        self.blocks = []

        for i in range(layers):
            block_layers = [
                nn.Conv2d((in_channels if i == 0 else out_channels),
                          out_channels, kernel_size, padding=kernel_size // 2),
                activation()
            ]
            if batchnorm:
                block_layers.insert(1, nn.BatchNorm2d(out_channels))
            block = nn.Sequential(*block_layers)
            self.blocks.append(block)

        self.main = nn.Sequential(*self.blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run layer"""
        results: Tensor = self.main(inputs)
        return results
