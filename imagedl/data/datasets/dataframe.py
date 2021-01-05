import pandas as pd

from .abstract import AbstractDataset, DataType, Transform


class FrameDataset(AbstractDataset):
    def __init__(self, df: pd.DataFrame, columns=None,
                 transform: Transform = None):
        super().__init__(transform=transform)
        if columns is None:
            self.df = df
        elif isinstance(columns, int) or isinstance(columns, str):
            self.df = df.loc[:, columns:]
        elif isinstance(columns, tuple):
            self.df = df.loc[:, columns[0]:columns[1]]
        self.columns = self.df.columns
        self.data = self.df.to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> DataType:
        return self._apply_transform(self.data[idx])
