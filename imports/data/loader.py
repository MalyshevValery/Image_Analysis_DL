"""All-in-one loader for image tasks"""
import re
from itertools import chain
from typing import List, Tuple, Iterable, Dict, Callable, Sequence, Union

import numpy as np

from imports.utils.types import to_seq, seq_apply, OMArray
from .storages import AbstractStorage

TrainValTest = Tuple[List[str], List[str], List[str]]
_FunType = Callable[[Sequence[AbstractStorage]], Sequence[np.ndarray]]
StorageType = Union[AbstractStorage, Sequence[AbstractStorage]]


class Loader:
    """Loader of data for different image tasks (currently only Semantic
    segmentation is supported)

    Loader is a middleware between storage and generators. It helps to unite
    data from different storages to create bundles of input and ground truth
    data which can be passed to generators;

    :param input_: Storage or Storages for input
    :param output_: Storage or Storages for output
    """

    def __init__(self, input_: StorageType, output_: StorageType):
        self.__input = input_
        self.__output = output_

        self.__tuple_input = to_seq(input_)
        self.__tuple_output = to_seq(output_)

        keys = self.__tuple_input[0].keys
        for storage in chain(self.__tuple_input, self.__tuple_output):
            keys = keys.intersection(tuple(storage.keys))
        self.__keys = list(keys)

    def split(self, train_val_test: Tuple[float, float, float],
              pattern: str = None) -> TrainValTest:
        """Splits indices on three groups to create training, test and
        validation sets.

        :param train_val_test: tuple or list of three elements with sum of 1,
        which contains fractures of whole set for train/validation/test split
        :param pattern: Apply this pattern to keys to get entities for split.
            This regex pattern must have one group catch.
        :returns shuffled keys for train , validation, test split
        """
        if sum(train_val_test) > 1:
            raise ValueError('Split', train_val_test, 'is greater than 1')
        if len(train_val_test) != 3:
            raise ValueError('Split', train_val_test, 'must have 3 elements')

        keys = np.array(self.keys)
        np.random.shuffle(keys)
        inverse_idx = np.array([])
        if pattern is None:
            train_val_test_counts = (
                    np.array(train_val_test) * len(keys)).astype(int)
        else:
            matches = [re.match(pattern, k) for k in keys]
            if None in matches:
                raise ValueError('Not all keys matched the expression')
            groups = [m.group(1) for m in matches if m is not None]
            keys_to_split, inverse_idx = np.unique(groups, return_inverse=True)
            train_val_test_counts = (
                    np.array(train_val_test) * len(keys_to_split)).astype(int)

        train_sep = train_val_test_counts[0]
        val_sep = train_val_test_counts[1] + train_sep
        test_sep = train_val_test_counts[2] + val_sep

        if pattern is None:
            train_keys = keys[:train_sep]
            val_keys = keys[train_sep:val_sep]
            test_keys = keys[val_sep:test_sep]
        else:
            shuffled_gid = np.arange(0, np.max(inverse_idx) + 1)
            np.random.shuffle(shuffled_gid)

            train_range = shuffled_gid[:train_sep]
            train_keys = keys[[k in train_range for k in inverse_idx]]
            val_range = shuffled_gid[train_sep:val_sep]
            val_keys = keys[[k in val_range for k in inverse_idx]]
            test_range = shuffled_gid[val_sep:test_sep]
            test_keys = keys[[k in test_range for k in inverse_idx]]
        return train_keys, val_keys, test_keys

    @property
    def keys(self) -> List[str]:
        """Getter for keys"""
        return self.__keys

    @property
    def input_shape(self) -> Iterable[Tuple[int, ...]]:
        """Returns input shape of data"""
        key = self.__keys[0]
        return (storage[key].shape for storage in to_seq(self.__input))

    def get_input(self, batch_keys: List[str]) -> OMArray:
        """Returns elements from input storages with specified keys"""
        return seq_apply(self.__input, Loader.__key_group(batch_keys))

    def get_output(self, batch_keys: List[str]) -> OMArray:
        """Returns elements from output storages with specified keys"""
        return seq_apply(self.__output, Loader.__key_group(batch_keys))

    def to_json(self) -> Dict[str, object]:
        """Returns JSON config for loader"""
        return {
            'input': [s.to_json() for s in to_seq(self.__input)],
            'output': [s.to_json() for s in to_seq(self.__output)]
        }

    @staticmethod
    def __key_group(keys: List[str]) -> _FunType:
        def group_fun(arr: Sequence[AbstractStorage]) -> Sequence[np.ndarray]:
            """Function returns list where every element is batch from one
                Storage"""
            return [np.array([s[key] for key in keys]) for s in arr]

        return group_fun
