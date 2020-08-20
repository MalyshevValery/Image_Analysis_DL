import glob
import os
from pathlib import Path

import numpy as np

from .abstract import AbstractDataset, DataType, Transform


class NumpyDataset(AbstractDataset):
    def __init__(self, glob_: Path, add_info=False,
                 filename_transforms=None, transform: Transform = None):
        super().__init__(transform=transform)
        self.paths = glob.glob(glob_)
        self.filenames = [os.path.basename(f) for f in self.paths]
        self.add_info = add_info

        self.info = {}
        if filename_transforms is not None:
            for k, v in filename_transforms.items():
                self.info[k] = np.array([v(f) for f in self.filenames])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> DataType:
        data = np.load(str(self.paths[idx]))
        if not self.add_info:
            return self._apply_transform(data)
        else:
            info = {}
            for k, v in self.info.items():
                info[k] = v[idx]
            return self._apply_transform((data, info))
