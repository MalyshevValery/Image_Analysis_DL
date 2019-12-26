"""Ignore region extension"""
import copy

import numpy as np
import skimage.morphology as morph

from .abstract import AbstractExtension


class IgnoreRegionExtension(AbstractExtension):
    """This extension creates empty region on all channels of provided mask
    based on morphological operations performed on one mask channel"""

    def __init__(self, radius=3, channel=1):
        """Constructor

        :param radius: radius of morphological element (disk)
        :param channel: channel with main ground truth for mask
        """
        self._morph_element = morph.disk(radius)
        self._channel = channel
        self._radius = radius

    @classmethod
    def type(cls):
        """Returns type of this extension"""
        return "ignore_region"

    def __call__(self, data):
        if data.shape[2] != 2:
            raise Exception('Only two data can be used with ignore label')
        ignore = np.zeros(data.shape[:-1], dtype=np.float32)
        ignore += morph.binary_dilation(data[:, :, self._channel],
                                        selem=self._morph_element) - data[:, :, self._channel]
        ignore += data[:, :, self._channel] - morph.binary_erosion(data[:, :, self._channel], selem=self._morph_element)
        data *= (1 - ignore[:, :, np.newaxis])
        return data

    def to_json(self):
        """Returns JSON config of IgnoreRegionExtension"""
        return {
            'type': IgnoreRegionExtension.type(),
            'radius': self._radius,
            'channel': self._channel
        }

    @staticmethod
    def from_json(json):
        """Returns IgnoreRegionExtension specified in json argument"""
        config = copy.deepcopy(json)
        if config.get('type', None) != IgnoreRegionExtension.type():
            raise ValueError('Invalid type ' + config.get('type') + ' for SplitMaskExtension')
        del config['type']

        return IgnoreRegionExtension(**config)

    @classmethod
    def allowed(cls):
        """IgnoreRegion can be applied only to masks"""
        return 'mask'
