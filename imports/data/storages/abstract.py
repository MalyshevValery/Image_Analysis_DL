"""Abstract Storage class"""
from abc import abstractmethod

from imports.jsonserializable import JSONSerializable


class AbstractStorage(JSONSerializable):
    """Storage class with declared methods required by storage"""

    def __init__(self, keys=None, mode='r'):
        """Constructor

        :param keys: keys for this storage
        :param mode: r to read, w to save data
        """
        if mode != 'r' and mode != 'w':
            raise ValueError('mode must be one of [r,w]')
        if mode == 'r' and keys is None:
            raise ValueError('Keys must be set in read mode')
        self._keys = keys
        self._mode = mode

    @abstractmethod
    def __getitem__(self, item):
        if self._mode != 'r':
            raise ValueError('Get item can be used only in read mode')
        pass

    def __len__(self):
        return len(self._keys)

    def get_keys(self):
        """Getter for key values"""
        return self._keys

    def save_array(self, keys, array):
        """Saves array to this storage (by default call save_single in a loop)

        :param keys: keys to save with
        :param array: array of data to save
        """
        if self._mode != 'w':
            raise ValueError('Save can be used only in write mode')
        if len(keys) != len(array):
            raise ValueError('Len of keys and array does not match')
        for i, key in enumerate(keys):
            self.save_single(key, array[i])

    @abstractmethod
    def save_single(self, key, data):
        """Method to save single data object into storage

        :param key: key for save
        :param data: data to save
        :return:
        """
        if self._mode != 'w':
            raise ValueError('Save can be used only in write mode')
        pass

    @classmethod
    @abstractmethod
    def type(cls):
        """Returns type of storage"""
        return 'abstract'
