"""HDF5 storage"""
from typing import Set, Dict

import numpy as np

from .abstract import AbstractStorage, ExtensionType


class MockStorage(AbstractStorage):
    """Storage for tests, emits same value for all given keys

    :param val: Value that will be returned on every __getitem__ call
    :param keys: Set of keys
    :param writable: True to write in dataset
    """

    def __init__(self, val: np.ndarray, keys: Set[str], writable: bool = False,
                 extensions: ExtensionType = None):
        self.__val = val.copy()
        super().__init__(keys, extensions, writable)

    def __getitem__(self, item: str) -> np.ndarray:
        """Returns item from dataset"""
        super().__getitem__(item)
        return self._apply_extensions(self.__val)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves single object to data storage"""
        print(f'Key {key} saved')

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Storage"""
        if self.__extensions is None:
            extensions = None
        else:
            extensions = [ext.to_json() for ext in self.__extensions]
        return {
            'type': 'mock',
            'extensions': extensions
        }
