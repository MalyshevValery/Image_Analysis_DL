"""Abstract Storage class"""
import enum
from abc import abstractmethod
from typing import Set, Union, List

import numpy as np
from ordered_set import OrderedSet


class Mode(enum.Enum):
    """Enum for READ and WRITE storage mods"""
    READ = 0,
    WRITE = 0


KeySet = Union[Set[str], OrderedSet]


class AbstractStorage:
    """Storage class with declared methods required by storage"""

    def __init__(self, keys: KeySet, mode: Mode = Mode.READ):
        """Constructor

        :param keys: keys for this storage
        :param mode: Mode.READ for read Mode.WRITE for write
        """
        if mode == Mode.READ and keys is None:
            raise ValueError('Keys must be set in read mode')
        self._keys = keys.copy()
        self._mode = mode

    @abstractmethod
    def __getitem__(self, item: str) -> np.ndarray:
        if self._mode is Mode.WRITE:
            raise ValueError('Write mode')
        return np.array([])

    def __len__(self) -> int:
        if self._keys is None:
            raise ValueError('Keys are None')
        return len(self._keys)

    @property
    def keys(self) -> KeySet:
        """Getter for key values"""
        if self._keys is None:
            raise ValueError('Keys are None')
        return self._keys

    def save_array(self, keys: List[str], array: np.ndarray) -> None:
        """Saves array to this storage (by default call save_single in a loop)

        :param keys: keys to save with
        :param array: array of data to save
        :raises ValueError if wrong mode or lengths of keys and array do not
            match
        """
        # if self._mode != Mode.WRITE:
        #     raise ValueError('Save can be used only in write mode')
        for i, key in enumerate(keys):
            self.save_single(key, array[i])

    @abstractmethod
    def save_single(self, key: str, data: np.ndarray) -> None:
        """Method to save single data object into storage

        :param key: key for save
        :param data: data to save
        """
        # if self._mode != Mode.WRITE:
        #     raise ValueError('Save can be used only in write mode')
        pass

    @classmethod
    @abstractmethod
    def type(cls) -> str:
        """Returns type of storage"""
        raise NotImplementedError
