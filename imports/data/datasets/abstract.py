"""Abstract Dataset"""
from abc import abstractmethod
from typing import Union, Callable, Tuple, Mapping

from torch import Tensor
from torch.utils.data import Dataset

DataType = Union[Tensor, Tuple[Tensor, ...], Mapping[str, Tensor]]
Transform = Callable[[DataType], DataType]


class AbstractDataset(Dataset):
    """Abstract dataset with transforms"""

    def __init__(self, transform: Transform = None):
        self.__transform = transform

    @abstractmethod
    def __getitem__(self, idx: int) -> DataType:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def _apply_transform(self, data: DataType) -> DataType:
        """Applies transformation to data"""
        if self.__transform is None:
            return data
        else:
            return self.__transform(data)
