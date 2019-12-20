"""Transforms data in [0, ?] to [0, X] of specified type"""
import copy

import numpy as np

from .abstract import AbstractExtension

ALLOWED_TYPES = {
    'float32': np.float32,
    'uint8': np.uint8
}


class TypeScaleExtension(AbstractExtension):
    """Extensions which allows type change and scaling of data"""

    def __init__(self, src_max=None, target_type=np.uint8, target_max=None):
        self._target_type = target_type
        self._src_max = src_max
        self._target_max = target_max if target_max is not None else np.iinfo(target_type).max

    def __call__(self, data):
        """Apply scale and type transform"""
        scaled_data = data.astype(np.float32) / self._src_max
        return (scaled_data * self._target_max).astype(self._target_type)

    def to_json(self):
        """Returns JSON config for this extension"""
        return {
            'type': TypeScaleExtension.type(),
            'src_max': self._src_max,
            'target_type': self._target_type.__name__,
            'target_max': self._target_max
        }

    @staticmethod
    def from_json(json):
        """Creates extension from json"""
        config = copy.deepcopy(json)
        if config.get('type', None) != TypeScaleExtension.type():
            raise ValueError('Invalid type ' + config.get('type') + ' for TypeScaleExtension')
        del config['type']

        if 'target_type' in config:
            if config['target_type'] not in ALLOWED_TYPES:
                raise ValueError(config['target_type'] + ' is not nalowed type for TypeScaleExtension')
            config['target_type'] = ALLOWED_TYPES[config['target_type']]
        return TypeScaleExtension(**config)

    @classmethod
    def type(cls):
        """Returns type of this image"""
        return 'type_scale'

    @classmethod
    def allowed(cls):
        """TypeScaleExtension is allowed to any data"""
        return 'all'
