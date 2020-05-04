"""Abstract Storage class"""
from abc import abstractmethod
from typing import Union, Callable, Tuple, Mapping

from torch.utils.data import Dataset
from torch import Tensor

DatasetType = Union[Tensor, Tuple[Tensor, ...], Mapping[str, Tensor]]


class AbstractDataset(Dataset):
    """Abstract dataset with transforms"""

    def __init__(self, transform: Callable[[DatasetType], DatasetType] = None):
        self.__transform = transform

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetType:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
