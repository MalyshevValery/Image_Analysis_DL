"""Lambda Dataset"""
from typing import Callable

from .abstract import AbstractDataset, DataType, Transform


class MapDataset(AbstractDataset):
    """
    Dataset which takes another dataset and maps its values to other with given
    function *func*

    :param ds: Parent Dataset
    :param func: Function to transform Parent Dataset values
    :param transform: Data Transforms
    """

    def __init__(self, ds: AbstractDataset,
                 func: Callable[[DataType], DataType],
                 transform: Transform = None):
        super().__init__(transform=transform)
        self.__ds = ds
        self.__func = func

    def __len__(self) -> int:
        return len(self.__ds)

    def __getitem__(self, idx: int) -> DataType:
        item = self.__func(self.__ds[idx])
        return self._apply_transform(item)
