"""Decorators for loaders to enhance their work"""
import numpy as np
from skimage.morphology import disk, binary_erosion, binary_dilation


def add_ignore(cls, radius=3, channel=1):
    """Decorator to add ignore labels to mask

    :param cls: Loader class
    :param radius: Radius of morphological element. Width of ignore label border will be about 2 * radius
    :param channel: Main ground truth channel
    """
    old_get_mask = cls.get_mask

    def wrapper(self, i):
        """Wrapper for mask"""
        mask = old_get_mask(self, i)
        if mask.shape[2] != 2:
            raise Exception('Only two channel mask can be used with ignore label')
        if getattr(self, '__morph_element', None) is None:
            setattr(self, '__morph_element', disk(radius))
        ignore = np.zeros(mask.shape[:-1], dtype=np.float32)
        ignore += binary_dilation(mask[:, :, channel], selem=self.__morph_element) - mask[:, :, channel]
        ignore += mask[:, :, channel] - binary_erosion(mask[:, :, channel], selem=self.__morph_element)
        mask *= (1 - ignore[:, :, np.newaxis])
        return mask

    cls.get_mask = wrapper
    return cls
