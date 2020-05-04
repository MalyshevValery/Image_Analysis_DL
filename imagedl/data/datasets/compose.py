"""Composed Dataset"""
from typing import Union, Tuple, Mapping

from .abstract import AbstractDataset, DataType, Transform

Composition = Union[
    Tuple[AbstractDataset, ...],
    Mapping[str, AbstractDataset]
]


class ComposeDataset(AbstractDataset):
    """
    Dataset which composes several datasets of same length in one

    :param datasets: Datasets to compose
    :param transform: Data Transform
    """

    def __init__(self, datasets: Composition, transform: Transform = None):
        super().__init__(transform=transform)
        self.__datasets = datasets
        if isinstance(self.__datasets, tuple):
            seq_datasets = self.__datasets
        elif isinstance(self.__datasets, Mapping):
            seq_datasets = tuple(self.__datasets.values())
        else:
            raise TypeError(f'Unknown type: {self.__datasets.__class__}')
        self.__len = len(seq_datasets[0])
        for ds in seq_datasets:
            if self.__len != len(ds):
                raise ValueError('Different lengths of datasets')

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, idx: int) -> DataType:
        if isinstance(self.__datasets, tuple):
            data: DataType = tuple(ds[idx] for ds in self.__datasets)
        elif isinstance(self.__datasets, Mapping):
            data = {k: ds[idx] for k, ds in self.__datasets.items()}
        else:
            raise ValueError('Datasets are not tuple or dict')
        return self._apply_transform(data)
