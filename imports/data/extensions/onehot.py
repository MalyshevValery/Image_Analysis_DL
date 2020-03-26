from typing import Dict

import numpy as np

from .abstract import AbstractExtension


class OneHotExtension(AbstractExtension):
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
