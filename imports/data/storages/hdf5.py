"""HDF5 storage"""
import copy

import h5py

from .abstract import AbstractStorage


class HDF5Storage(AbstractStorage):
    """Storage to work with .h5 files"""

    def __init__(self, filename, dataset_name, mode='r', shape=None, dtype=None):
        self._filename = filename
        self._dataset_name = dataset_name

        file = h5py.File(filename, 'a')
        if mode == 'r':
            self._dataset = file[dataset_name]
        else:
            if shape is None or dtype is None:
                raise ValueError('shape or value has not been provided for write storage')
            if dataset_name in file:
                print('Replacing dataset ' + dataset_name)
                del file[dataset_name]
            self._dataset = file.create_dataset(dataset_name, shape=shape, dtype=dtype)
            self._counter = 0
        super().__init__(set(range(len(self._dataset))), mode=mode)

    def __getitem__(self, item):
        """Returns item from dataset"""
        super().__getitem__(item)
        return self._dataset[item]

    def save_array(self, keys, array):
        """Saves array to h5 file"""
        super().save_array(keys, array)

        self._dataset[list(range(self._counter, self._counter + len(keys)))] = array
        self._counter += len(keys)

    def save_single(self, key, data):
        """Saves single object to data storage"""
        self._dataset[self._counter] = data
        self._counter += 1

    @classmethod
    def type(cls):
        """Returns type of this storage"""
        return 'hdf5'

    def to_json(self):
        """Returns JSON config of this class"""
        return {
            'type': HDF5Storage.type(),
            'filename': self._filename,
            'dataset_name': self._dataset_name
        }

    @staticmethod
    def from_json(json, mode='r'):
        """Creates object of this class from JSON"""
        config = copy.deepcopy(json)
        if config.pop('type', None) != HDF5Storage.type():
            raise ValueError('Type ' + json['type'] + ' is invalid type for HDF5Storage')
        filename = config.pop('filename', None)
        dataset_name = config.pop('dataset_name', None)
        return HDF5Storage(filename, dataset_name, mode=mode, **config)
