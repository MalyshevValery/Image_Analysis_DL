"""Class for mask channel splitting"""
import copy

import numpy as np

from .abstract import AbstractExtension


class SplitMaskExtension(AbstractExtension):
    """Extension that split one channel mask into several channel mask according to provided codes

    Provided codes can be list or integer. In case of integer codes are setup as [0...n-1] if n is provided integer.
    This extension finds all values on mask equal to code from list and use these values to generate new binary mask for
    specified code and inserts it as one channel of new multi-channel mask.
    """

    def __init__(self, codes=2):
        """Constructor

        :param codes: codes to identify different class labels in one channel, can be list of such codes or int,
        int code identify number of labels on mask based starting from zero, as [0...(n-1)]
        """
        if not isinstance(codes, list):
            codes = list(range(codes))
        self._codes = codes

    @classmethod
    def allowed(cls):
        """This extension is allowed only for masks"""
        return 'mask'

    @classmethod
    def type(cls):
        """Returns name of this extension"""
        return 'split_mask'

    def __call__(self, data):
        """Provided mask is split into some channels that are merged to create one multi-channel mask according to
        provided codes

        Extension can be applied only to one channel masks (grayscale)"""
        if data.shape[-1] != 1:
            raise ValueError('Data should be grayscale image with single channel - (n, m, 1)')
        data = data[:, :, 0]
        new_data = np.zeros([*data.shape, len(self._codes)], dtype=data.dtype)
        for i, c in enumerate(self._codes):
            new_data[..., i] = data == c
        return new_data

    def to_json(self):
        """Returns JSON config of SplitMaskExtension"""
        return {
            'type': SplitMaskExtension.type(),
            'codes': self._codes,
        }

    @staticmethod
    def from_json(json):
        """Returns SplitMaskExtension specified in json argument"""
        config = copy.deepcopy(json)
        if config.get('type', None) != SplitMaskExtension.type():
            raise ValueError('Invalid type ' + config.get('type') + ' for SplitMaskExtension')
        del config['type']

        return SplitMaskExtension(**config)
