"""Wrapper for Augmentation mapping to data in generator"""
import enum
from typing import Tuple, Callable, List, Dict, Iterator

import numpy as np
from albumentations import BasicTransform


class IO(enum.Enum):
    """Input or output variable"""
    INPUT = 'i',
    OUTPUT = 'o'


_DataId = Tuple[IO, int]
_AL = List[np.ndarray]
_AT = Tuple[np.ndarray]
_DataType = Tuple[_AL, _AL]
_IterType = Tuple[_AT, _AT]
_NAMES = ('image', 'mask', 'bboxes', 'keypoints')


class AugmentationMap:
    """
    This class transforms given input and output due to given configuration

    :param image: Image identifier in data
    :param mask: Mask identifier in data
    :param bboxes: Bounding boxes identifier in data
    :param keypoints: Keypoints identifier in data
    """

    def __init__(self, image: _DataId = None,
                 mask: _DataId = None, bboxes: _DataId = None,
                 keypoints: _DataId = None):
        gen = zip(_NAMES, [image, mask, bboxes, keypoints])
        self._dict = {e[0]: e[1] for e in gen if e[1] is not None}

    def __call__(self, input_: _AL, output_: _AL,
                 augmentation: BasicTransform) -> None:
        """Applies given augmentation to given data. This operation mutates
            given NumPy arrays.

        Note that input and output tuples have to contain arrays of shape
            [batch_size, ...] because they will be unrolled

        :param input_: Input tuple of numpy arrays
        :param output_: Output tuple of numpy arrays
        :param augmentation: Augmentation to apply
        :return: Augmented input and output
        """
        output_dict: Dict[str, _AL] = {k: [] for k in self._dict.keys()}
        data_iter: Iterator[_IterType] = zip(zip(*input_), zip(*output_))
        for input_entry, output_entry in data_iter:
            map_function = AugmentationMap.__mapper(input_entry, output_entry)
            input_dict = {k: map_function(*v) for k, v in self._dict.items()}
            for k, v in augmentation(**input_dict).items():
                output_dict[k].append(v)

        for k, v in output_dict.items():
            io, index = self._dict[k]
            if io is IO.INPUT:
                input_[index] = np.stack(v)
            elif io is IO.OUTPUT:
                output_[index] = np.stack(v)

    @staticmethod
    def __mapper(input_: _AT,
                 output_: _AT) -> Callable[[IO, int], np.ndarray]:
        def map_function(io: IO, index: int) -> np.ndarray:
            """Function which maps IO and index to data"""
            if io is IO.INPUT:
                return input_[index]
            elif io is IO.OUTPUT:
                return output_[index]

        return map_function

    def to_json(self) -> Dict[str, object]:
        """Returns JSON config for Augmentation data mapping"""
        return {k: (str(io), index) for k, (io, index) in self._dict.items()}
