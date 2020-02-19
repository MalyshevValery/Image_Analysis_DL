"""Abstract extension"""
from abc import abstractmethod

import numpy as np


class AbstractExtension:
    """Abstract extension class

    Abstract extensions is used in Loader to extend default capabilities of
    Storage methods. Mostly extensions should be used to change data format,
    normalize or slightly change it. For augmentation please consult
    albumentations package and Sequence, which is used as main source of data
    for Keras training and test processes.
    """

    @classmethod
    @abstractmethod
    def type(cls) -> str:
        """Returns string which identifies extension"""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply transformation to data and returns transformed data"""
        raise NotImplementedError
