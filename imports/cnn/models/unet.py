from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Dropout, Activation, BatchNormalization, concatenate
from tensorflow.keras import Model


def Conv2DBlock(input_tensor, n_filters, n_layers=2, activation='relu', kernel_size=3, batchnorm=True):
    x = input_tensor
    for i in range(n_layers):
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer="he_normal", padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
    return x


def UNet(input_shape, n_filters=16, dropout=0.5, batchnorm=True, kernel_size=3, activation='relu', n_conv_layers=2):
    """U-Net architecture

    :param input_shape: shape of one input image
    :param n_filters: number of filters in convolutions. This number will progressively increase with 1,2,4,8,16 multiplier
    :param dropout: Dropout parameter
    :param batchnorm: If batchnorm iwll be applied
    :param kernel_size: Size of convolutional kernel
    :param activation: Activation function
    :param n_conv_layers: Number of convolutoinal layers in each convolutoinal/deconvolutional block
    :return: Keras model for U-net
    """
    input_img = Input(input_shape)

    # contracting path
    c1 = Conv2DBlock(input_img, n_filters=n_filters * 1, kernel_size=kernel_size, activation=activation,
                     batchnorm=batchnorm, n_layers=n_conv_layers)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = Conv2DBlock(p1, n_filters=n_filters * 2, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = Conv2DBlock(p2, n_filters=n_filters * 4, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = Conv2DBlock(p3, n_filters=n_filters * 8, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = Conv2DBlock(p4, n_filters=n_filters * 16, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = Conv2DBlock(u6, n_filters=n_filters * 8, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = Conv2DBlock(u7, n_filters=n_filters * 4, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = Conv2DBlock(u8, n_filters=n_filters * 2, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = Conv2DBlock(u9, n_filters=n_filters * 1, kernel_size=kernel_size, activation=activation, batchnorm=batchnorm,
                     n_layers=n_conv_layers)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs], name='U-Net')
    return model
