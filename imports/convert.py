"""Data format conversion"""
import copy

from imports.jsonserializable import JSONSerializable
from .data import Loader, AbstractStorage, storage_factory


class ConvertWrapper(JSONSerializable):
    """Class to convert data from one format to another"""

    def __init__(self, loader: Loader, storage_to: AbstractStorage, loader_method='image'):
        self._loader = loader
        self._storage_to = storage_to
        self._get_method = ConvertWrapper.loader_method(loader, loader_method)
        self._method_name = loader_method

    def convert(self):
        """Converts data form loader to storage"""
        for key in self._loader.get_keys():
            self._storage_to.save_single(key, self._get_method(key))

    def to_json(self):
        """Saves configuration to JSON"""
        return {
            'loader': self._loader.to_json(),
            'storage_to': self._storage_to.to_json(),
            'loader_method': self._method_name
        }

    @staticmethod
    def from_json(json):
        """Restores object from JSON configuration"""
        config = copy.deepcopy(json)
        loader = Loader.from_json(config.pop('loader', None))
        storage_config = config.pop('storage_to', None)
        storage_config.update(ConvertWrapper.additional_config(loader, storage_config['type'],
                                                               config.get('loader_method', 'image')))
        storage_to = storage_factory(storage_config, 'w')
        return ConvertWrapper(loader, storage_to, **config)

    @staticmethod
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

    @staticmethod
    def additional_config(loader: Loader, storage_type, method_name='image'):
        """This method returns additional configuration required for different storage

        :param loader: Loader instance
        :param storage_type:
        :param method_name:
        :return:
        """
        add_conf = {}
        if storage_type == 'hdf5':
            example = ConvertWrapper.loader_method(loader, method_name)(loader.get_keys()[0])
            add_conf['shape'] = (len(loader.get_keys()), *example.shape)
            add_conf['dtype'] = example.dtype
        return add_conf
