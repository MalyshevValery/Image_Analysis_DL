import copy

import albumentations as ab
from albumentations.core.serialization import SERIALIZABLE_REGISTRY

from imports.jsonserializable import JSONSerializable

NAME_TO_REGISTRY = dict((k.split('.')[-1], k) for k in SERIALIZABLE_REGISTRY.keys())


class AlbumentationsWrapper(JSONSerializable):
    """Wrapper for albumentations serialization procedure"""

    @staticmethod
    def to_json(aug):
        """Turns albumentations to JSON"""
        raw_dict = ab.to_dict(aug)
        transform_dict = raw_dict['transform']
        return AlbumentationsWrapper.__deep_apply(transform_dict, '__class_fullname__', 'transform',
                                                  lambda s: s.split('.')[-1])

    @staticmethod
    def from_json(json):
        """Creates albumentations from JSON"""
        if isinstance(json, list):
            json = {"transform": "Compose", "transforms": json}
        copied_dict = copy.deepcopy(json)
        transform_dict = AlbumentationsWrapper.__deep_apply(copied_dict, 'transform', '__class_fullname__',
                                                            lambda s: NAME_TO_REGISTRY[s])
        return ab.from_dict({'__version__': ab.__version__, 'transform': transform_dict})

    @staticmethod
    def __deep_apply(to_change, from_key, to_key, transform_function):
        if isinstance(to_change, dict):
            if from_key in to_change:
                to_change[to_key] = transform_function(to_change[from_key])
                del to_change[from_key]

            for k in to_change.keys():
                to_change[k] = AlbumentationsWrapper.__deep_apply(to_change[k], from_key, to_key, transform_function)
            return to_change
        elif isinstance(to_change, list):
            return [AlbumentationsWrapper.__deep_apply(val, from_key, to_key, transform_function) for val in to_change]
        else:
            return to_change
