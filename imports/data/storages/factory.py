"""Factory method for storage"""
from .directory import DirectoryStorage
from .hdf5 import HDF5Storage


def storage_factory(json, mode='r'):
    """Creates proper storage from specified json"""
    storage_type = json['type']
    if storage_type == DirectoryStorage.type():
        return DirectoryStorage.from_json(json, mode)
    elif storage_type == HDF5Storage.type():
        return HDF5Storage.from_json(json, mode)
    else:
        raise ValueError('Type ' + json['type'] + " is unknown type")
