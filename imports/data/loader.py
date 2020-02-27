"""All-in-one loader for image tasks"""
from itertools import chain
from typing import List, Tuple, Iterable, Dict

import numpy as np

from .storages import AbstractStorage

TrainValTest = Tuple[List[str], List[str], List[str]]


class Loader:
    """Loader of data for different image tasks (currently only Semantic
    segmentation is supported)

    Loader is a middleware between storage and generators. It helps to unite
    data from different storages to create bundles of input and ground truth
    data which can be passed to generators;

    :param input_: Storage or Storages for input
    :param output_: Storage or Storages for output
    """

    def __init__(self, input_: Tuple[AbstractStorage, ...],
                 output_: Tuple[AbstractStorage, ...]):

        keys = input_[0].keys
        for storage in chain(input_, output_):
            keys = keys.intersection(tuple(storage.keys))
        self._keys = list(keys)  # To ensure order

        self._input = input_
        self._output = output_

    def split(self, train_val_test: Tuple[float, float, float]) -> TrainValTest:
        """Splits indices on three groups to create training, test and
        validation sets.

        :param train_val_test: tuple or list of three elements with sum of 1,
        which contains fractures of whole set for train/validation/test split
        :returns shuffled keys for train , validation, test split
        """
        if sum(train_val_test) > 1:
            raise ValueError('Split', train_val_test, 'is greater than 1')
        if len(train_val_test) != 3:
            raise ValueError('Split', train_val_test, 'must have 3 elements')
        np.random.shuffle(self._keys)
        train_val_test_counts = (
                np.array(train_val_test) * len(self._keys)).astype(int)
        train_count = train_val_test_counts[0]
        test_count = train_val_test_counts[2]

        train_keys = self._keys[:train_count]
        val_keys = self._keys[train_count:-test_count]
        test_keys = self._keys[-test_count:]
        return train_keys, val_keys, test_keys

    def keys(self) -> List[str]:
        """Getter for keys"""
        return self._keys

    def get_input_shape(self) -> Iterable[Tuple[int]]:
        """Returns input shape of data"""
        key = self._keys[0]
        return (storage[key].shape for storage in self._input)

    def get_input(self, batch_keys: List[str]) -> Iterable[np.ndarray]:
        """Returns elements from input storages with specified keys"""
        return [np.array([s[key] for key in batch_keys]) for s in self._input]

    def get_output(self, batch_keys: List[str]) -> Iterable[np.ndarray]:
        """Returns elements from output storages with specified keys"""
        return [np.array([s[key] for key in batch_keys]) for s in self._output]

    def to_json(self) -> Dict[str, object]:
        """Returns JSON config for loader"""
        return {
            'input': [s.to_json() for s in self._input],
            'output': [s.to_json() for s in self._output]
        }
