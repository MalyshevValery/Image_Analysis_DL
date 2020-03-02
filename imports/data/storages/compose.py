"""Composed storage class"""
from typing import Dict, Set, Sequence, Union

import numpy as np

from imports.utils import to_seq
from .abstract import AbstractStorage, ExtensionType

_StorageType = Union[AbstractStorage, Sequence[AbstractStorage]]


class ComposeStorage(AbstractStorage):
    """Storage that is Union of other storages

    Keys format is '<i>/<key>' where i is index of required storage and key is
    key in that storage. Same refers to saving mode.

    :param storages: Storages to compose
    :param extensions: Extensions to apply
    :param writable: True if writable
    """

    def __init__(self, storages: _StorageType, extensions: ExtensionType = None,
                 writable: bool = True):
        keys: Set[str] = set()
        self.__storages = to_seq(storages)
        for i, storage in enumerate(self.__storages):
            keys.update(f'{i}/{key}' for key in storage.keys)
        super().__init__(keys, extensions, writable)

    def __getitem__(self, item: str) -> np.ndarray:
        idx = item.find('/')
        return self.__storages[int(item[:idx])][item[idx + 1:]]

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves single data entry with given key"""
        if not self.writable:
            raise ValueError('Not writable')
        idx = key.find('/')
        self.__storages[int(key[:idx])].save_single(key[idx + 1:], data)

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration of this Storage"""
        return {
            'type': 'composed',
            'storages': [storage.to_json() for storage in self.__storages],
            'extensions': self._extensions_json()
        }
