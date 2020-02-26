"""UNet architecture for semantic segmentation"""
from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, concatenate
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Softmax

from .conv2dblock import Conv2DBlock


def UNet(input_shape: Tuple[int, ...], out_channels: int = 1,
         n_filters: int = 16, dropout: float = 0.5, batchnorm: bool = True,
         kernel_size: int = 3, activation: str = 'relu',
         n_conv_layers: int = 2) -> Model:
    """U-Net architecture

    :param input_shape: shape of one input image
    :param out_channels: number of output channels
    :param n_filters: number of filters in convolutions.
        This number will increase with 1,2,4,8,16 multiplier
    :param dropout: Dropout parameter
    :param batchnorm: If batchnorm will be applied
    :param kernel_size: Size of convolution kernel
    :param activation: Activation function
    :param n_conv_layers: Number of convolution layers in each
        convolution/deconvolution block
    :return: Keras model for U-net
    """
    input_img = Input(input_shape)

    # contracting path
    c1 = Conv2DBlock(n_filters=n_filters * 1, kernel_size=kernel_size,
                     activation=activation,
                     batchnorm=batchnorm, n_layers=n_conv_layers)(input_img)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = Conv2DBlock(n_filters=n_filters * 2, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = Conv2DBlock(n_filters=n_filters * 4, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = Conv2DBlock(n_filters=n_filters * 8, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(p3)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = Conv2DBlock(n_filters=n_filters * 16, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(p4)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(
        c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = Conv2DBlock(n_filters=n_filters * 8, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(u6)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(
        c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = Conv2DBlock(n_filters=n_filters * 4, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(u7)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(
        c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = Conv2DBlock(n_filters=n_filters * 2, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(u8)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(
        c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = Conv2DBlock(n_filters=n_filters * 1, kernel_size=kernel_size,
                     activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)(u9)

    if out_channels > 1:
        outputs = Conv2D(out_channels, (1, 1))(c9)
        outputs = Softmax()(outputs)
    else:
        outputs = Conv2D(out_channels, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img], outputs=[outputs], name='UNet')
    return model
