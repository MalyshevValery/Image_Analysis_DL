"""Factory method for storage"""
from .directory import DirectoryStorage


def storage_factory(json, mode='r'):
    """Creates proper storage from specified json"""
    storage_type = json['type']
    if storage_type == DirectoryStorage.type():
        return DirectoryStorage.from_json(json, mode)
    else:
        raise ValueError('Type ' + json['type'] + " is unknown type")
