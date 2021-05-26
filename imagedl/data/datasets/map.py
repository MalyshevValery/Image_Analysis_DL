"""Lambda Dataset"""
from typing import Callable, Optional, Sized

from torch.utils.data.dataset import Dataset, T_co


class MapDataset(Dataset[T_co]):
    """
    Dataset which takes another dataset and maps its values to another with
    given function *func*

    :param ds: Parent Dataset
    :param func: Function to transform Parent Dataset values
    """

    def __init__(self, ds: Dataset[T_co],
                 func: Callable[[T_co], T_co]):
        super().__init__()
        self.__ds = ds
        self.__func = func

    def __len__(self) -> Optional[int]:
        if isinstance(self.__ds, Sized):
            return len(self.__ds)
        else:
            return None

    def __getitem__(self, idx: int) -> T_co:
        return self.__func(self.__ds[idx])
