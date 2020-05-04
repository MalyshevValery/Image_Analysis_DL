"""Mock dataset"""
from .abstract import AbstractDataset, DataType, Transform


class MockStorage(AbstractDataset):
    """
    Dataset which return same value on every index

    :param value: Value to return every time
    :param total: Length of this dataset
    :param transform: Data Transform
    """

    def __init__(self, value: DataType, total: int,
                 transform: Transform = None):
        self.__val = value
        self.__total = total
        super().__init__(transform=transform)

    def __len__(self) -> int:
        return self.__total

    def __getitem__(self, idx: int) -> DataType:
        return self._apply_transform(self.__val)
