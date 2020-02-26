"""HDF5 storage"""
from typing import Set

import numpy as np

from .abstract import AbstractStorage, Mode, ExtensionType


class MockStorage(AbstractStorage):
    """Storage for tests, emits same value for all given keys

    :param val: Value that will be returned on every __getitem__ call
    :param keys: Set of keys
    :param mode: READ/WRITE mode
    """

    def __init__(self, val: np.ndarray, keys: Set[str], mode: Mode = Mode.READ,
                 extensions: ExtensionType = None):
        self.__val = val.copy()
        super().__init__(keys, mode, extensions)

    def __getitem__(self, item: str) -> np.ndarray:
        """Returns item from dataset"""
        super().__getitem__(item)
        return self._apply_extensions(self.__val)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves single object to data storage"""
        print('Key %s saved' % key)

    @classmethod
    def type(cls) -> str:
        """Returns type of this storage"""
        return 'mock'
