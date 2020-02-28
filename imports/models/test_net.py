"""Test architecture"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow_core.python.keras.layers import Conv2DTranspose


def TestNet() -> Model:
    """Test architecture with two inputs and two outputs"""
    input_img1 = Input([256, 256, 1])

    c1 = Conv2D(32, 3, activation='relu', padding='same')(input_img1)
    p1 = MaxPooling2D((2, 2))(c1)
    d1 = Dropout(0.1)(p1)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(d1)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)
    p5 = MaxPooling2D((2, 2))(c5)

    dc1 = Conv2DTranspose(128, 3, activation='relu',
                          strides=2, padding='same')(p4)
    c6 = Conv2D(64, 3, activation='relu', padding='same')(dc1)

    dc2 = Conv2DTranspose(16, 3, activation='relu',
                          strides=2, padding='same')(c6)
    c7 = Conv2D(8, 3, activation='relu', padding='same')(dc2)

    dc3 = Conv2DTranspose(4, 3, activation='relu',
                          strides=2, padding='same')(c7)
    c8 = Conv2D(1, 3, activation='sigmoid', padding='same')(dc3)

    flat = Flatten()(p5)
    d1 = Dense(64, activation='relu')(flat)
    d2 = Dense(16, activation='relu')(d1)
    res = Dense(1, activation='sigmoid')(d2)
    model = Model(inputs=input_img1, outputs=c8,
                  name='TestNet')
    return model
