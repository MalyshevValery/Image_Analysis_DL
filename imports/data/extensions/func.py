"""Abstract extension"""
from typing import Dict, Callable

import numpy as np
from .abstract import AbstractExtension


class LambdaExtension(AbstractExtension):
    """Pass function which will be applied to every data entry"""
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
        self.__func = func

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply transformation to data and returns transformed data"""
        return self.__func(data)

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Extension"""
        return {
            'type': 'lambda'
        }
