"""Wrapper for Augmentation mapping to data in generator"""
import enum
from typing import Tuple, Callable, List, Dict, Iterator, Sequence

import numpy as np
from albumentations import BasicTransform

from imports.utils.types import to_seq, apply_as_seq, OMArray


class IO(enum.Enum):
    """Input or output variable"""
    INPUT = 'i',
    OUTPUT = 'o'


_DataId = Tuple[IO, int]
_AL = List[np.ndarray]
_AT = Tuple[np.ndarray]
_IterType = Tuple[_AT, _AT]
_FuncType = Callable[[Sequence[np.ndarray]], Sequence[np.ndarray]]
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

    def __call__(self, input_: OMArray, output_: OMArray,
                 augmentation: BasicTransform) -> Tuple[OMArray, OMArray]:
        """Applies given augmentation to given data. This operation mutates
            given NumPy arrays.

        Note that input and output tuples have to contain arrays of shape
            [batch_size, ...] because they will be unrolled

        :param input_: Input tuple of numpy arrays
        :param output_: Output tuple of numpy arrays
        :param augmentation: Augmentation to apply
        :return: Augmented input and output
        """
        result: Dict[str, _AL] = {k: [] for k in self._dict.keys()}
        data_iter: Iterator[_IterType] = zip(zip(*to_seq(input_)),
                                             zip(*to_seq(output_)))
        for input_entry, output_entry in data_iter:
            map_function = AugmentationMap.__mapper(input_entry, output_entry)
            input_dict = {k: map_function(*v) for k, v in self._dict.items()}
            for k, v in augmentation(**input_dict).items():
                result[k].append(v)

        new_input = apply_as_seq(input_, self.__get_data(result, IO.INPUT))
        new_output = apply_as_seq(output_, self.__get_data(result, IO.OUTPUT))
        return new_input, new_output

    def to_json(self) -> Dict[str, object]:
        """Returns JSON config for Augmentation data mapping"""
        return {k: (str(io), index) for k, (io, index) in self._dict.items()}

    @staticmethod
    def __mapper(input_: _AT,
                 output_: _AT) -> Callable[[IO, int], np.ndarray]:
        def map_function(io: IO, index: int) -> np.ndarray:
            """Function which maps IO and index to data"""
            if io is IO.INPUT:
                return to_seq(input_)[index]
            elif io is IO.OUTPUT:
                return to_seq(output_)[index]

        return map_function

    def __get_data(self, dict_: Dict[str, _AL], io: IO) -> _FuncType:
        def set_data(seq: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
            """Changes data entries on augmented ones"""
            seq = list(seq)
            for k, v in dict_.items():
                io_, index = self._dict[k]
                if io_ is io and index in range(len(seq)):
                    seq[index] = np.stack(v)
            return seq

        return set_data
