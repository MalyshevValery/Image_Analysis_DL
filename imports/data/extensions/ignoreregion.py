"""Ignore region extension"""
from typing import Dict

import numpy as np
import skimage.morphology as morph

from .abstract import AbstractExtension


class IgnoreRegionExtension(AbstractExtension):
    """This extension creates empty region on all channels of provided mask,
    based on morphological operations performed on one mask channel

    This extensions selects region in which empty space will be created by
    making region from one dilation and one erosion of specified radius. Ignore
    region is added to add some level of freedom to ML predictions

    **IMPORTANT** If chosen loss function penalize for high rate of false
    positives this approach won't work (jaccard and dice losses penalize for
    false predictions when cross-entropy doesn't)

    Apply to - **MASK**

    :param radius: radius of morphological element (disk)
    :param channel: channel with main ground truth for mask
    """

    def __init__(self, radius: int = 3, channel: int = 1):
        self.__radius = radius
        self.__channel = channel
        self.__morph_element = morph.disk(self.__radius)

    @classmethod
    def type(cls) -> str:
        """Returns type of this extension"""
        return 'ignore_region'

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Creates empty region which will be ignored due to lack of any ground
        truth data on this region"""
        if data.shape[2] != 2:
            raise ValueError('This extension work on only two dimensional data')
        ignore = np.zeros(data.shape[:-1], dtype=np.float32)

        channel = data[:, :, self.__channel]
        ignore += morph.binary_dilation(channel, selem=self.__morph_element)
        ignore -= morph.binary_erosion(channel, selem=self.__morph_element)
        data *= (1 - ignore[:, :, np.newaxis])
        return data

    def to_json(self) -> Dict[str, object]:
        """JSON configuration for this Extension"""
        return {
            'type': 'ignore_region',
            'radius': self.__radius,
            'channel': self.__channel
        }
