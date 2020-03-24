"""Transforms data in [0, ?] to [0, X] of specified type"""
from typing import Dict

import numpy as np

from .abstract import AbstractExtension


class OneHotExtension(AbstractExtension):
    """Type scale extensions allows to change type of array data as well as
    linearly scale data to specified region.

    There are two main types for now -- float32 and uint8. All scales takes
    place from 0 to specified value or default one. TypeScaleExtension allows
    to load data form image files and after that turn it to floating point
    values as well as vice-versa. It's worth to note that for training data
    it's better to use ToFloat augmentation due to better performance of
    augmentation on uint8 data.

    :param src_max: maximum value on source data (if None every data entry will
        be scaled to its own maximum value)
    :param target_type: string identifier of required target datatype
    :param target_max: maximum value for target. If None and target_type is
        integer like target_max will be maximum available value for this type
    """

    def __init__(self, n_classes: int):
        self.__n_classes = n_classes

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply scale and type transform"""
        arr = np.zeros(self.__n_classes)
        arr[data] = 1.0
        return arr

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Extension"""
        return {
            'type': 'one_hot',
            'n_classes': self.__n_classes
        }
