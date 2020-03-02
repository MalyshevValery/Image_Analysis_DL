"""Abstract Storage class"""
from abc import abstractmethod
from typing import Set, Union, List, Dict, Optional, Sequence

import numpy as np

from imports.data.extensions import AbstractExtension
from imports.utils import to_seq
from imports.utils.torderedset import TOrderedSet

KeySet = Union[Set[str], TOrderedSet[str]]
ExtensionType = Union[AbstractExtension, Sequence[AbstractExtension]]


class AbstractStorage:
    """Storage class with declared methods required by storage

    :param keys: keys for this storage
    :param writable: True to allow writing to storage
    :param extensions: Extensions to apply to this storage
    """

    def __init__(self, keys: KeySet, extensions: ExtensionType = None,
                 writable: bool = False):
        if not writable and keys is None:
            raise ValueError('Keys must be set in read mode')
        self.__keys = keys.copy()
        self.__writable = writable
        if extensions is None:
            self.__extensions = None
        else:
            self.__extensions = to_seq(extensions)

    @abstractmethod
    def __getitem__(self, item: str) -> np.ndarray:
        raise NotImplementedError()

    def __len__(self) -> int:
        if self.__keys is None:
            raise ValueError('Keys are None')
        return len(self.__keys)

    @property
    def keys(self) -> Set[str]:
        """Getter for key values"""
        if self.__keys is None:
            raise ValueError('Keys are None')
        return self.__keys.copy()

    @property
    def writable(self) -> bool:
        """Returns writable property"""
        return self.__writable

    def save_array(self, keys: List[str], array: np.ndarray) -> None:
        """Saves array to this storage (by default call save_single in a loop)

        :param keys: keys to save with
        :param array: array of data to save
        :raises ValueError if wrong mode or lengths of keys and array do not
            match
        """
        if not self.__writable:
            raise ValueError('Not writable')
        for i, key in enumerate(keys):
            self.save_single(key, array[i])

    def _apply_extensions(self, data: np.ndarray) -> np.ndarray:
        cur = data
        if self.__extensions is None:
            return cur
        else:
            for ext in self.__extensions:
                cur = ext(cur)
            return cur

    def _add_keys(self, keys: Union[str, Sequence[str]]) -> None:
        """Adds key to set of keys"""
        self.__keys.update(to_seq(keys))

    def _extensions_json(self) -> Optional[List[object]]:
        if self.__extensions is None:
            return None
        else:
            return [ext.to_json() for ext in self.__extensions]

    @abstractmethod
    def save_single(self, key: str, data: np.ndarray) -> None:
        """Method to save single data object into storage.

        Don't forget to check for writable parameter.

        :param key: key for save
        :param data: data to save
        """
        raise NotImplementedError()

    @abstractmethod
    def to_json(self) -> Dict[str, object]:
        """Returns configuration of Storage in JSON format"""
        raise NotImplementedError()
