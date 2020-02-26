"""Parts for NN models"""
from typing import Mapping

from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D


class Conv2DBlock(layers.Layer):
    """
    Keras layer which includes some 2D convolution layers of same kernel and
    channel size, activation function and batch normalization parameter

    :param n_filters: number of output channels
    :param n_layers: number of convolutions in block
    :param activation: activations function
    :param kernel_size: size of kernel
    :param batchnorm: True if batch normalization should be applied
    """

    def __init__(self, n_filters: int, n_layers: int = 2,
                 activation: str = 'relu', kernel_size: int = 3,
                 batchnorm: bool = True):
        super(Conv2DBlock, self).__init__()
        self.layers = []
        self.config = {
            'n_layers': n_layers,
            'n_filters': n_filters,
            'activation': activation,
            'kernel_size': kernel_size,
            'batchnorm': batchnorm
        }

        for i in range(n_layers):
            block = [Conv2D(filters=n_filters,
                            kernel_size=(kernel_size, kernel_size),
                            kernel_initializer="he_normal",
                            padding="same")]
            if batchnorm:
                block.append(BatchNormalization())
            block.append(Activation(activation))
            self.layers.append(block)

    def call(self, inputs: Tensor, **kwargs: object) -> Tensor:
        """Run layer"""
        x = inputs
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x

    def get_config(self) -> Mapping[str, object]:
        """Config of layer"""
        return self.config
