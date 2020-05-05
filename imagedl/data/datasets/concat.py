"""Concat dataset"""
from typing import Tuple

import numpy as np

from .abstract import AbstractDataset, DataType, Transform


class ConcatDataset(AbstractDataset):
    """Dataset to concatenate other datasets in one"""

    def __init__(self, datasets: Tuple[AbstractDataset, ...],
                 transform: Transform = None):
        super().__init__(transform=transform)
        self.__datasets = datasets
        self.__lengths = np.array([len(ds) for ds in self.__datasets])
        self.__separators = np.cumsum(self.__lengths)

    def __len__(self) -> int:
        return int(self.__separators[-1])

    def __getitem__(self, idx: int) -> DataType:
        next_id = np.where(self.__separators - idx > 0)
        ds = self.__datasets[next_id - 1]
        return ds[idx - self.__separators[next_id - 1]]
