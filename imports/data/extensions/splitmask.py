"""Class for mask channel splitting"""
from typing import Union, List, Dict

import numpy as np

from .abstract import AbstractExtension


class SplitMaskExtension(AbstractExtension):
    """Extension that split one channel mask into several channel mask according
        to provided codes

    Provided codes can be list or integer. In case of integer codes are setup as
    [0...n-1] if n is provided integer. This extension finds all values on mask
    equal to code from list and use these values to generate new binary mask for
    specified code and inserts it as one channel of new multi-channel mask.

    :param codes: codes to identify different class labels in one channel, can
        be list of such codes or int, int code identify number of labels on mask
        based starting from zero, as [0...(n-1)]
    """

    def __init__(self, codes: Union[int, List[int]] = 2):
        if isinstance(codes, int):
            self.__codes = list(range(codes))
        else:
            self.__codes = list(codes)

    @classmethod
    def type(cls) -> str:
        """Returns name of this extension"""
        return 'split_mask'

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Provided mask is split into some channels that are merged to create
        one multi-channel mask according to
        provided codes

        Extension can be applied only to one channel masks (grayscale)

        :param data: NumPy array with data to process
        :raises ValueError if DataShape is not (n, m, 1)
        """
        if len(data.shape) == 3:
            if data.shape[-1] != 1:
                raise ValueError('Data shape have to be (n, m, 1)')
            data = data[:, :, 0]
        new_data = np.zeros([*data.shape, len(self.__codes)], dtype=data.dtype)
        for i, c in enumerate(self.__codes):
            new_data[..., i] = data == c
        return new_data

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this Extension"""
        return {
            'type': 'split_mask',
            'codes': self.__codes
        }
