"""SubDataset"""
from typing import Sequence

from .abstract import AbstractDataset, DataType, Transform


class SubDataset(AbstractDataset):
    """
    Dataset which takes data entries from a subset of another dataset entries
    
    :param source: Source dataset
    :param indexes: Subset of indexes
    :param transform: Data transform
    """

    def __init__(self, source: AbstractDataset, indexes: Sequence[int],
                 transform: Transform = None):
        super().__init__(transform=transform)
        self.source = source
        self.__indexes = indexes

    def __getitem__(self, idx: int) -> DataType:
        return self._apply_transform(self.source[self.__indexes[idx]])

    def __len__(self) -> int:
        return len(self.__indexes)
