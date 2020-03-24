"""HDF5 storage"""
from typing import List, Type, Tuple, Dict, Sequence

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
    :param replace: True if writable and existing dataset have to be replaced
        **This parameter cause keys dataset replacement too**
    :param shape: Shape of dataset to create
    :param dtype: Type of data in that dataset
    """

    def __init__(self, filename: str, dataset_name: str,
                 extensions: ExtensionType = None,
                 writable: bool = False, replace: bool = False,
                 shape: Tuple[int, ...] = None,
                 dtype: Type[np.generic] = None):

        self.__filename = filename
        if dataset_name == _keys_dataset:
            raise ValueError(f'{dataset_name} is equal kto key dataset name')
        self.__dataset_name = dataset_name

        self.__file = h5py.File(filename, 'a')
        if writable and (dataset_name not in self.__file or replace):
            error_str = 'Shape or Value was not provided for write storage'
            if shape is None or dtype is None:
                raise ValueError(error_str)
            if dataset_name in self.__file:
                print('Replacing dataset ' + dataset_name)
                del self.__file[dataset_name]
            self.__dataset = self.__file.create_dataset(dataset_name, shape,
                                                        dtype)
        else:
            self.__dataset = self.__file[dataset_name]

        keys: TOrderedSet[str] = TOrderedSet()
        if _keys_dataset in self.__file:
            if replace:
                del self.__file[_keys_dataset]
                self.__create_key_dataset()
            else:
                self.__keys_dataset = self.__file[_keys_dataset]
                keys = TOrderedSet(self.__keys_dataset)
                if '' in keys:
                    keys.remove('')

        elif writable:
            self.__create_key_dataset()
        else:
            raise ValueError('No Keys in read-only storage')

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
        new_keys = []
        indexes = []
        for i, k in enumerate(keys):
            if k in self.keys:
                idx = self.__ordered_keys.index(k)
                self.__dataset[idx] = array[i]
            else:
                indexes.append(i)
                new_keys.append(k)
        start = len(self.keys)
        if start + len(new_keys) > self.__dataset.shape[0]:
            raise ValueError('Size of dataset is not enough for writing')
        self.__dataset[start:start + len(new_keys)] = array[indexes]
        self.__keys_dataset[start:start + len(new_keys)] = new_keys
        self._add_keys(new_keys)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """Saves single object to data storage"""
        self.save_array([key], data[np.newaxis])

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Storage"""
        return {
            'type': 'hdf5',
            'filename': self.__filename,
            'dataset': self.__dataset_name,
            'extensions': self._extensions_json()
        }

    def _add_keys(self, keys: Sequence[str]) -> None:
        super()._add_keys(keys)
        self.__ordered_keys.update(keys)

    def __create_key_dataset(self) -> None:
        stype = h5py.string_dtype()
        size = len(self.__dataset)
        self.__keys_dataset = self.__file.create_dataset(_keys_dataset,
                                                         (size,), stype)

    def close(self) -> None:
        """Closes file in hdf5 context"""
        self.__file.close()
