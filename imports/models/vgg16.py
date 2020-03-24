from typing import Optional, Sequence

from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, \
    GlobalAveragePooling2D


def customVGG16(top: Sequence[int] = None, weights: Optional[str] = 'imagenet',
                input_shape: Sequence[int] = None, pooling: str = None,
                classes: int = 1000,
                classifier_activation: str = 'softmax',
                ave: bool = False) -> Model:
    """Returns VGG16 with custom layers

    :param top: None to include VGG16 or sequence of FC channels
    :param weights: None, 'imagenet' or path to weights
    :param input_shape: Tuple with input shape
    :param pooling: If include_top is False then None, 'avg', 'max',
    :param classes: Number of classes
    :param classifier_activation: Classifier activation on the top
        (only if top is not None)
    """
    vgg16 = VGG16(include_top=(top is None), weights=weights,
                  input_shape=input_shape, classes=classes, pooling=pooling)
    if top is None:
        return vgg16
    else:
        input_ = Input(shape=input_shape, name='image_input')
        output = vgg16(input_)
        if ave:
            fc = GlobalAveragePooling2D()(output)
        else:
            fc = Flatten()(output)
        for channels in top:
            fc = Dense(channels, activation='relu')(fc)
        custom_output = Dense(classes, activation=classifier_activation)(fc)
        return Model(inputs=(input_,), outputs=(custom_output,))
