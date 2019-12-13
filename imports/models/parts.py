"""Parts for NN models"""
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras import layers


class Conv2DBlock(layers.Layer):
    """Block of some 2D convolution blocks of same size and kernel"""

    def __init__(self, n_filters, n_layers=2, activation='relu', kernel_size=3, batchnorm=True):
        """Creates block of some 2D convolution blocks of same size and kernel

        :param n_filters: number of output channels
        :param n_layers: number of convolutions in block
        :param activation: activations function
        :param kernel_size: size of kernel
        :param batchnorm: True if batchnorm should be applied
        :return: output of last layer
        """
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
            block = [Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                            padding="same")]
            if batchnorm:
                block.append(BatchNormalization())
            block.append(Activation(activation))
            self.layers.append(block)

    def call(self, inputs, **kwargs):
        """Run layer"""
        x = inputs
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x

    def get_config(self):
        """Config of layer"""
        return self.config