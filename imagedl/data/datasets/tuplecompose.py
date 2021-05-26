"""Dataset composing several datasets values into tuples"""
from typing import Tuple, Sized, Optional

from torch.utils.data.dataset import Dataset, T_co


class TupleComposeDataset(Dataset[Tuple[T_co, ...]]):
    """
    Dataset which composes several datasets of same length in one
    and returns tuples
    """

    def __init__(self, *datasets: Dataset[T_co]):
        super().__init__()
        self.__datasets = datasets
        self.__len = None
        for d in self.__datasets:
            if isinstance(d, Sized):
                self.__len = len(d)
                break

    def __len__(self) -> Optional[int]:
        return self.__len

    def __getitem__(self, idx: int) -> Tuple[T_co, ...]:
        return tuple(ds[idx] for ds in self.__datasets)
