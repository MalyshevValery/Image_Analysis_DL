from typing import Dict

import numpy as np

from .abstract import AbstractExtension


class ChannelExtension(AbstractExtension):
    def __init__(self, axis=-1):
        self.__axis = axis

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply scale and type transform"""
        return np.expand_dims(data, self.__axis)

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Extension"""
        return {
            'type': 'channel',
            'axis': self.__axis
        }
