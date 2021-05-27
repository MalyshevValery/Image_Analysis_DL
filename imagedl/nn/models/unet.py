"""UNet architecture for semantic segmentation"""
from typing import Type

from torch import nn, Tensor, cat

from imagedl.nn.models.blocks import Conv2dBlock


class UNet(nn.Module):
    """U-Net architecture

    :param n_channels: Number of input image channels
    :param n_classes: Number of classes in Network
    :param dropout: Dropout parameter
    :param batchnorm: Whether batchnorm must be applied
    :param kernel_size: Size of convolution kernel
    :param activation: Activation Module
    :param n_layers: Number of convolution layers in each
        convolution/deconvolution block
    """

    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.5,
                 batchnorm: bool = True, kernel_size: int = 3,
                 activation: Type[nn.Module] = nn.ReLU, n_layers: int = 2):
        super(UNet, self).__init__()
        self.__n_channels = n_channels
        self.__n_classes = n_classes
        self.__dropout = dropout
        self.__batchnorm = batchnorm
        self.__activation = activation
        self.__n_layers = n_layers

        self.inc = Conv2dBlock(n_channels, 64)
        self.down1 = UNet.__down_layer(64, 128, dropout, n_layers, activation,
                                       kernel_size, batchnorm)
        self.down2 = UNet.__down_layer(128, 256, dropout, n_layers, activation,
                                       kernel_size, batchnorm)
        self.down3 = UNet.__down_layer(256, 512, dropout, n_layers, activation,
                                       kernel_size, batchnorm)
        self.down4 = UNet.__down_layer(512, 1024, dropout, n_layers, activation,
                                       kernel_size, batchnorm)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_block = Conv2dBlock(1024, 512, n_layers, activation,
                                     kernel_size, batchnorm)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_block = Conv2dBlock(512, 256, n_layers, activation,
                                     kernel_size, batchnorm)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_block = Conv2dBlock(256, 128, n_layers, activation,
                                     kernel_size, batchnorm)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_block = Conv2dBlock(128, 64, n_layers, activation,
                                     kernel_size, batchnorm)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run UNet"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1_block(cat([self.up1(x5), x4], dim=1))
        x = self.up2_block(cat([self.up2(x), x3], dim=1))
        x = self.up3_block(cat([self.up3(x), x2], dim=1))
        x = self.up4_block(cat([self.up4(x), x1], dim=1))

        result: Tensor = self.out(x)
        return result

    @staticmethod
    def __down_layer(in_channels: int, out_channels: int, dropout: float,
                     n_layers: int, activation: Type[nn.Module],
                     kernel_size: int, batchnorm: bool) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.MaxPool2d(2),
            Conv2dBlock(in_channels, out_channels, n_layers, activation,
                        kernel_size, batchnorm)
        )
