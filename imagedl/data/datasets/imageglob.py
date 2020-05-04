"""Image Glob Dataset"""
import glob
import os

from skimage import io

from .abstract import AbstractDataset, DataType, Transform


class ImageGlobDataset(AbstractDataset):
    """
    Scikit-image based dataset which loads images from a directory

    :param query: Glob to retrieve image paths
    :param include_filename: If True __getitem__ returns dict with image and
    filename
    :param transform: Data Transform
    """

    def __init__(self, query: str, include_filename: bool = False,
                 transform: Transform = None):
        super().__init__(transform=transform)
        self.__paths = sorted(glob.glob(query))
        self.__include_filename = include_filename

    def __len__(self) -> int:
        return len(self.__paths)

    def __getitem__(self, idx: int) -> DataType:
        path = self.__paths[idx]
        image = io.imread(path)
        image = self._apply_transform(image)
        if self.__include_filename:
            return {'image': image, 'filename': os.path.basename(path)}
        else:
            return image
