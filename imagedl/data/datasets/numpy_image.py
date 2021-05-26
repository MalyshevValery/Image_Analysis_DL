"""Dataset for loading npy or image files"""
import glob
import os
from typing import List, Mapping, Callable, Any, Tuple, Union

import numpy as np
from skimage.io import imread
from torch.utils.data.dataset import Dataset

image_ext = ['.png', '.jpg', '.jpeg']
Transform = Callable[[str], str]
Transforms = Mapping[str, Transform]
ReturnType = Union[np.ndarray, Tuple[np.ndarray, Mapping[str, Any]]]


class NumpyImageDataset(Dataset[ReturnType]):
    """Dataset for extracting images or npy files by glob

    :param glob_: Globbing string for getting paths to files
    :param add_info: If true dataset returns tuple of file data and additional
    info, only file data otherwise

    :param filename_transforms: Dictionary of functions which transforms
    filename into some useful info which is stored in info field and can be
    attached to data items if add_info is set

    :param path_transforms: Same as filename_transforms but these functions are
    applied to full paths of data files
    """

    def __init__(self, glob_: str, add_info: bool = False,
                 filename_transforms: Transforms = None,
                 path_transforms: Transforms = None):
        super().__init__()
        self.paths: List[str] = glob.glob(glob_)
        self.filenames = [os.path.basename(f) for f in self.paths]
        self.add_info = add_info

        self.info = {}
        if filename_transforms is not None:
            for k, v in filename_transforms.items():
                self.info[k] = np.array([v(f) for f in self.filenames])
        if path_transforms is not None:
            for k, v in path_transforms.items():
                self.info[k] = np.array([v(f) for f in self.paths])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> ReturnType:
        data: np.ndarray = np.array(0)
        if self.paths[idx].endswith('.npy'):
            data = np.load(str(self.paths[idx]))
        elif any(self.paths[idx].endswith(ext) for ext in image_ext):
            data = imread(self.paths[idx])
        if not self.add_info:
            return data
        else:
            info = {}
            for k, v in self.info.items():
                info[k] = v[idx]
            return data, info
