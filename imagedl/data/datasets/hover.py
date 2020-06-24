"""HDF5 Dataset"""

import os
from pathlib import Path

import numpy as np

from .abstract import AbstractDataset, DataType, Transform


class HoverDataset(AbstractDataset):
    """
    Dataset for HoVerNet
    """

    def __init__(self, data_dir: Path, transform: Transform = None):
        super().__init__(transform=transform)
        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)
        self.filenames = np.sort(np.array(self.filenames))

        self.wsis = np.array([f.split('-')[-1][:-4] for f in self.filenames])
        self.ids = np.array([f.split('-')[0] for f in self.filenames])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> DataType:
        data = np.load(str(self.data_dir / self.filenames[idx]))
        return self._apply_transform(data)
