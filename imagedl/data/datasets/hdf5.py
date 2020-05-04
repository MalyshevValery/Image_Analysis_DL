"""HDF5 Dataset"""
from typing import Tuple

import h5py
from torch import from_numpy

from .abstract import AbstractDataset, DataType, Transform


class HDF5Dataset(AbstractDataset):
    """
    HDF5 Dataset

    :param filename: Filename for hdf5 dataset
    :param datasets_names: Names of datasets in HDF5 to load
    :param as_tuple: If True __getitem__ returns tuple of objects otherwise it
        returns dict with dataset names as keys
    """

    def __init__(self, filename: str,
                 datasets_names: Tuple[str, ...], as_tuple: bool = False,
                 transform: Transform = None):
        super().__init__(transform=transform)
        self.__filename = filename
        self.__names = datasets_names
        self.__as_tuple = as_tuple

        self.__file = h5py.File(filename, 'r')
        self.__datasets = [self.__file[name] for name in self.__names]
        self.__len = int(self.__datasets[0].shape[0])
        for ds in self.__datasets:
            if ds.shape[0] != self.__len:
                raise ValueError('Datasets in hdf5 have different length')

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, idx: int) -> DataType:
        if self.__as_tuple:
            data: DataType = tuple(
                from_numpy(ds[idx]) for ds in self.__datasets)
        else:
            data = {self.__names[i]: from_numpy(ds[idx]) for i, ds in
                    enumerate(self.__datasets)}
        return self._apply_transform(data)
