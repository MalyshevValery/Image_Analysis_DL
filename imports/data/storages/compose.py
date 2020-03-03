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
                 writable: bool = False):
        keys: Set[str] = set()
        self.__storages = to_seq(storages)

        storages_writable = all(s.writable for s in self.__storages)
        if writable and not storages_writable:
            raise ValueError('Storages inside are not writable')

        for i, storage in enumerate(self.__storages):
            keys.update(f'{i}/{key}' for key in storage.keys)
        super().__init__(keys, extensions, writable)

    def __getitem__(self, item: str) -> np.ndarray:
        idx = item.find('/')
        data = self.__storages[int(item[:idx])][item[idx + 1:]]
        return self._apply_extensions(data)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves single data entry with given key"""
        if not self.writable:
            raise ValueError('Not writable')
        idx = key.find('/')
        self.__storages[int(key[:idx])].save_single(key[idx + 1:], data)
        self._add_keys(key)

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration of this Storage"""
        return {
            'type': 'compose',
            'storages': [storage.to_json() for storage in self.__storages],
            'extensions': self._extensions_json()
        }
