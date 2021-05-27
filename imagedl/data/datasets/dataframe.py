"""Dataset which provides dataframe rows as items"""
from typing import Union, Tuple

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


class FrameDataset(Dataset[np.ndarray]):
    """Pandas dataframe dataset"""

    def __init__(self, df: pd.DataFrame,
                 columns: Tuple[Union[str, int]] = None):
        super().__init__()
        if columns is None:
            self.df = df
        elif isinstance(columns, tuple):
            self.df = df.loc[:, columns]
        self.columns = self.df.columns
        self.data = self.df.to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.array(self.data[idx])
