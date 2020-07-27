"""HDF5 Dataset"""
import glob
import pickle

import numpy as np
import sparse

from .abstract import AbstractDataset, DataType, Transform


class PandaDataset(AbstractDataset):
    """
    Dataset for HoVerNet
    """

    def __init__(self, glob_: str, transform: Transform = None):
        super().__init__(transform=transform)
        self.paths = glob.glob(glob_)
        self.filenames = [p.split('/')[-1][:-4] for p in self.paths]

        self.ids = np.array([f.split('_')[0] for f in self.filenames])
        self.gl1 = np.array([int(f.split('_')[1]) for f in self.filenames])
        self.gl2 = np.array([int(f.split('_')[2]) for f in self.filenames])
        self.isup = np.array([int(f.split('_')[3]) for f in self.filenames])

        self.gl1[self.gl1 > 0] -= 2
        self.gl2[self.gl2 > 0] -= 2

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> DataType:
        pth = self.paths[idx]
        if pth.endswith('npy'):
            data = np.load(str(self.paths[idx]))
        else:
            with open(pth, 'rb') as f:
                matrix = pickle.load(f)
                data = sparse.asnumpy(matrix)
        return self._apply_transform((data, (self.isup[idx], self.gl1[idx], self.gl2[idx])))
