"""Concat dataset"""
import numpy as np

from .abstract import AbstractDataset, DataType, Transform


class ConcatDataset(AbstractDataset):
    """Dataset to concatenate other datasets in one"""

    def __init__(self, *datasets: AbstractDataset,
                 transform: Transform = None):
        super().__init__(transform=transform)
        self.__datasets = datasets
        self.__lengths = np.array([len(ds) for ds in self.__datasets])
        self.__separators = np.cumsum(self.__lengths)

    def __len__(self) -> int:
        return int(self.__separators[-1])

    def __getitem__(self, idx: int) -> DataType:
        ds_id = np.where(self.__separators - idx > 0)[0][0]
        ds = self.__datasets[ds_id]
        if ds_id > 0:
            idx -= self.__separators[ds_id - 1]
        return self._apply_transform(ds[idx])
