"""Mock dataset"""
from torch.utils.data.dataset import Dataset, T_co


class MockDataset(Dataset[T_co]):
    """
    Dataset which return same value on every index

    :param value: Value to return every time
    :param total: Length of this dataset
    """

    def __init__(self, value: T_co, total: int):
        super().__init__()
        self.__val = value
        self.__total = total
        if self.__total <= 0:
            raise ValueError('Total must be greater than zero')

    def __len__(self) -> int:
        return self.__total

    def __getitem__(self, idx: int) -> T_co:
        return self.__val
