"""Storage which extracts classes from keys"""
import re
from typing import Dict

import numpy as np

from .abstract import AbstractStorage, KeySet, ExtensionType


class ClassKeyStorage(AbstractStorage):
    """
    Storage which transforms keys string into classes by matching them to
    given pattern and extracting 1st group

    Notes:
    - Class do not support writing mode
    - After initializing it is possible to pass keys not from initial keys set
        in __getitem__

    :param keys; Given keys to find classes range
    :param pattern: Pattern to match
    :param one_hot: True to make one-hot encoding, class int label otherwise
    :param extensions: Extensions to apply
    :param writable: True if writable
    """

    def __init__(self, keys: KeySet, pattern: str, one_hot: bool = False,
                 extensions: ExtensionType = None, writable: bool = False):
        self.__pattern = re.compile(pattern)
        self.__keys = keys
        self.__one_hot = one_hot

        if writable:
            raise ValueError('ClassKey can not be writable')

        matches = (self.__pattern.match(k) for k in keys)
        classes = (m.group(1) for m in matches if m is not None)
        unique_classes = np.unique([c for c in classes])
        self.__class_dict = {k: i for i, k in enumerate(sorted(unique_classes))}

        super().__init__(keys, extensions, writable)

    def __getitem__(self, item: str) -> np.ndarray:
        match = self.__pattern.match(item)
        if match is None:
            raise ValueError(f'Match of {item} is None')
        clazz = match.group(1)
        label = self.__class_dict[clazz]
        if not self.__one_hot:
            return self._apply_extensions(np.array(label, dtype=np.float32))
        else:
            one_hot = np.zeros(len(self.__class_dict))
            one_hot[label] = 1
            return self._apply_extensions(one_hot)

    def save_single(self, key: str, data: np.ndarray) -> None:
        """
        Saves single data entry to storage

        :raises NotImplementedError because ClassKeyStorage is not writable
        """
        raise NotImplementedError()

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this ClassKeyStorage"""
        return {
            'type': 'class_key',
            'class_dict': self.__class_dict,
            'pattern': self.__pattern.pattern,
            'extensions': self._extensions_json()
        }

    @property
    def class_dict(self) -> Dict[str, int]:
        """Returns class - label mapping"""
        return dict(self.__class_dict)
