"""Dataset composing several datasets values into tuples"""
from typing import Sized, Optional, Mapping

from torch.utils.data.dataset import Dataset, T_co


class MapComposeDataset(Dataset[Mapping[str, T_co]]):
    """
    Dataset which composes several datasets of same length in one
    and returns tuples
    """

    def __init__(self, datasets: Mapping[str, Dataset[T_co]]):
        super().__init__()
        self.__datasets = datasets
        self.__len = None
        for d in self.__datasets.values():
            if isinstance(d, Sized):
                self.__len = len(d)
                break

    def __len__(self) -> Optional[int]:
        return self.__len

    def __getitem__(self, idx: int) -> Mapping[str, T_co]:
        return {k: d[idx] for k, d in self.__datasets.items()}
