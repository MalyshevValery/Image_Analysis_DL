"""Conversion script"""
import argparse
import json
import os
import sys
from collections import Iterable

sys.path.append(os.path.dirname(__file__))
from imports.data import Loader, storage_factory


def loader_method(loader: Loader, method_name='image'):
    """Returns proper bounded loader method by provided key

    :param loader: Loader instance
    :param method_name: method for getting data (image or mask)
    :return:
    """
    if method_name == 'image':
        return loader.get_image
    elif method_name == 'mask':
        return loader.get_mask
    else:
        raise ValueError(method_name + ' is unknown loader method')


def additional_config(loader: Loader, storage_type, method):
    """This method returns additional configuration required for different storage

    :param loader: Loader instance
    :param storage_type: type of created storage
    :param method: loader method top get data
    :return:
    """
    add_conf = {}
    if storage_type == 'hdf5':
        example = method(loader.get_keys()[0])
        add_conf['shape'] = (len(loader.get_keys()), *example.shape)
        add_conf['dtype'] = example.dtype
    return add_conf


def convert(settings='settings.json'):
    """Converts data from one format to another"""
    with open(settings, 'r') as file:
        config = json.load(file)
    loader = Loader.from_json(config.pop('loader', None))
    tasks = config.pop('tasks', None)
    if not isinstance(tasks, Iterable):
        tasks = [tasks]
    for storage_config in tasks:
        print('Doing...', storage_config)
        method = loader_method(loader, storage_config.pop('loader_method', 'image'))
        storage_config.update(additional_config(loader, storage_config['type'], method))
        storage = storage_factory(storage_config, 'w')
        for key in loader.get_keys():
            storage.save_single(key, method(key))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('settings', help='File with settings for data conversion', nargs='?',
                      default='settings.json')
    parsed_args = args.parse_args(sys.argv[1:])

    settings_arg = parsed_args.settings

    if os.path.isfile(settings_arg):
        print('File -', settings_arg)
        convert(settings_arg)
