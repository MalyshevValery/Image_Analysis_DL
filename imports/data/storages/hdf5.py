"""HDF5 storage"""
from typing import List, Type, Tuple

import h5py
import numpy as np

from imports.utils.torderedset import TOrderedSet
from .abstract import AbstractStorage, Mode, ExtensionType

_keys_dataset = '__keys'


class HDF5Storage(AbstractStorage):
    """Storage to work with HDF5 (.h5) files

    :param filename: Filename of h5 file
    :param dataset_name: Name of dataset in h5 file
    :param mode: READ/WRITE mode
    :param dtype: Type of data in that dataset
    """

    def __init__(self, filename: str, dataset_name: str, mode: Mode = Mode.READ,
                 extensions: ExtensionType = None,
                 shape: Tuple[int, ...] = None,
                 dtype: Type[np.generic] = None):

        self._filename = filename
        if dataset_name == _keys_dataset:
            raise ValueError('%s is equal kto key dataset name' % dataset_name)
        self._dataset_name = dataset_name

        self._file = h5py.File(filename, 'a')
        if mode is Mode.WRITE:
            if shape is None or dtype is None:
                raise ValueError(
                    'Shape or Value has not been provided for write storage')
            if dataset_name in self._file:
                print('Replacing dataset ' + dataset_name)
                del self._file[dataset_name]
            self._dataset = self._file.create_dataset(dataset_name, shape,
                                                      dtype)
        elif mode is Mode.READ:
            self._dataset = self._file[dataset_name]

        keys: TOrderedSet[str] = TOrderedSet()
        if _keys_dataset in self._file:
            self._keys_dataset = self._file[_keys_dataset]
            keys = TOrderedSet(self._keys_dataset)
            keys.remove('')
        else:
            stype = h5py.string_dtype()
            size = len(self._dataset)
            self._keys_dataset = self._file.create_dataset(_keys_dataset, size,
                                                           stype)
        super().__init__(keys, mode, extensions)

    def __getitem__(self, item: str) -> np.ndarray:
        """Returns item from dataset"""
        super().__getitem__(item)

        idx = int(item)
        if isinstance(self._keys, TOrderedSet):
            idx = self._keys.index(item)
        return self._apply_extensions(self._dataset[idx])

    def save_array(self, keys: List[str], array: np.ndarray) -> None:
        """Saves array to h5 file"""
        super().save_array(keys, array)

        start = len(self._keys)
        self._dataset[start:start + len(keys)] = array
        self._keys_dataset[start:start + len(keys)] = keys
        self._keys.update(keys)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves single object to data storage"""
        self.save_array([key], data[np.newaxis])

    @classmethod
    def type(cls) -> str:
        """Returns type of this storage"""
        return 'hdf5'
