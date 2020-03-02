"""HDF5 storage"""
from typing import List, Type, Tuple, Dict

import h5py
import numpy as np

from imports.utils.torderedset import TOrderedSet
from .abstract import AbstractStorage, ExtensionType

_keys_dataset = '__keys'


class HDF5Storage(AbstractStorage):
    """Storage to work with HDF5 (.h5) files

    :param filename: Filename of h5 file
    :param dataset_name: Name of dataset in h5 file
    :param writable: True to make writable storage
    :param shape: Shape of dataset to create
    :param dtype: Type of data in that dataset
    """

    def __init__(self, filename: str, dataset_name: str,
                 extensions: ExtensionType = None,
                 writable: bool = False,
                 shape: Tuple[int, ...] = None,
                 dtype: Type[np.generic] = None):

        self.__filename = filename
        if dataset_name == _keys_dataset:
            raise ValueError(f'{dataset_name} is equal kto key dataset name')
        self.__dataset_name = dataset_name

        self.__file = h5py.File(filename, 'a')
        if writable:
            if shape is None or dtype is None:
                raise ValueError(
                    'Shape or Value has not been provided for write storage')
            if dataset_name in self.__file:
                print('Replacing dataset ' + dataset_name)
                del self.__file[dataset_name]
            self.__dataset = self.__file.create_dataset(dataset_name, shape,
                                                        dtype)
        else:
            self.__dataset = self.__file[dataset_name]

        keys: TOrderedSet[str] = TOrderedSet()
        if _keys_dataset in self.__file:
            self.__keys_dataset = self.__file[_keys_dataset]
            keys = TOrderedSet(self.__keys_dataset)
            if '' in keys:
                keys.remove('')
        else:
            stype = h5py.string_dtype()
            size = len(self.__dataset)
            self.__keys_dataset = self.__file.create_dataset(_keys_dataset,
                                                             size, stype)
        self.__ordered_keys = keys
        super().__init__(self.__ordered_keys, extensions, writable)

    def __getitem__(self, item: str) -> np.ndarray:
        """Returns item from dataset"""
        idx = self.__ordered_keys.index(item)
        return self._apply_extensions(self.__dataset[idx])

    def save_array(self, keys: List[str], array: np.ndarray) -> None:
        """Saves array to h5 file"""
        if not self.writable:
            raise ValueError('Not writable')
        start = len(self.keys)
        self.__dataset[start:start + len(keys)] = array
        self.__keys_dataset[start:start + len(keys)] = keys
        self._add_keys(keys)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves single object to data storage"""
        self.save_array([key], data[np.newaxis])

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Storage"""
        return {
            'type': 'hdf5',
            'filename': self.__filename,
            'dataset_name': self.__dataset_name,
            'extensions': self._extensions_json()
        }
