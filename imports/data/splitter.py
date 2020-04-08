"""Classes for data splitting and DataGenerator creation"""
import numpy as np
import re
from typing import NamedTuple, Tuple, List, Dict, Generator
from albumentations import BasicTransform, Compose

from .augmentationmap import AugmentationMap
from .generator import DataGenerator
from .loader import Loader

_TrainValTest = Tuple[float, float, float]


class Split(NamedTuple):
    """Tuple with train, validation and test DataGenerators"""
    train: DataGenerator
    val: DataGenerator
    test: DataGenerator

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this object"""
        return {
            'train': self.train.to_json(),
            'val': self.val.to_json(),
            'test': self.test.to_json()
        }


class Splitter:
    """Class which splits data into train validation and test sets

    :param loader: Loader object to get keys from
    :param batch_size: Batch size
    :param augmentation_train: Additional augmentations during train stage
    :param augmentation_all: Augmentations after augmentation_train that are
        applied on all stages
    :param augmentation_map: AugmentationMap object to select data for
        augmentation
    """

    def __init__(self, loader: Loader, batch_size: int = 1,
                 augmentation_train: BasicTransform = None,
                 augmentation_all: BasicTransform = None,
                 augmentation_map: AugmentationMap = None, ):
        self.__loader = loader
        self.__batch_size = batch_size

        self.__aug_all = augmentation_all
        self.__aug_map = augmentation_map
        self.__aug_composed = augmentation_train
        if augmentation_train and augmentation_all:
            self.__aug_composed = Compose([self.__aug_composed, self.__aug_all])
        elif self.__aug_all:
            self.__aug_composed = self.__aug_all

    def random_split(self, train_val_test: _TrainValTest,
                     pattern: str = '(.*)') -> Split:
        """Splits indices randomly on three groups to create training, test and
                validation sets.

        :param train_val_test: tuple or list of three elements with sum of 1,
            which contains fractures of whole set for train/validation/test
            split
        :param pattern: Apply this pattern to keys to get entities for split.
            This regex pattern must have one group catch. Default pattern
            matches whole keys.
        :returns Shuffled keys for train , validation, test split
        """
        if sum(train_val_test) > 1:
            raise ValueError('Split', train_val_test, 'is greater than 1')
        if len(train_val_test) != 3:
            raise ValueError('Split', train_val_test, 'must have 3 elements')

        keys = np.array(self.__loader.keys)
        np.random.shuffle(keys)
        split_keys = self.__split_keys(keys, np.array(train_val_test), pattern)
        return self.__keys_to_split(*split_keys)

    def k_fold(self, val: float, n_fold: int,
               pattern: str = '(.*)') -> Generator[Split, None, None]:
        """Returns n_fold splits according to K-Fold technique on shuffled keys

        :param val: Fracture of validation part in train fold
        :param n_fold: Number of folds
        :param pattern: Apply this pattern to keys to get entities for split.
            This regex pattern must have one group catch. Default pattern
            matches whole keys.
        """
        if not 0 < val < 1:
            raise ValueError(f'Validation frac {val} must be in (0,1) interval')
        if not 2 <= n_fold <= len(self.__loader.keys):
            raise ValueError(f'Number of folds must be in [1, len(keys)]')

        keys = np.array(self.__loader.keys)
        np.random.shuffle(keys)
        split = np.full(n_fold, 1/n_fold)
        tv_split = np.array([1 - val, val])
        split_keys = self.__split_keys(keys, split, pattern)

        for i in range(n_fold):
            test_keys = list(split_keys[i])
            tv_keys = [k for j, k in enumerate(split_keys) if j != i]
            tv_keys = np.concatenate(tv_keys)
            t_v_keys = self.__split_keys(tv_keys, tv_split, pattern)
            yield self.__keys_to_split(list(t_v_keys[0]),
                                       list(t_v_keys[1]),
                                       test_keys)

    def __keys_to_split(self, train_keys: List[str], val_keys: List[str],
                        test_keys: List[str]) -> Split:
        train_gen = DataGenerator(train_keys, self.__loader, self.__batch_size,
                                  self.__aug_map, self.__aug_composed, True)
        val_gen = DataGenerator(val_keys, self.__loader, self.__batch_size,
                                self.__aug_map, self.__aug_all, False)
        test_gen = DataGenerator(test_keys, self.__loader, self.__batch_size,
                                 self.__aug_map, self.__aug_all, False)
        return Split(train_gen, val_gen, test_gen)

    @staticmethod
    def __get_groups(keys: np.ndarray,
                     pattern: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns unique groups acquired by matching pattern to keys
        and inverse index for keys"""
        matches = [re.match(pattern, k) for k in keys]
        if None in matches:
            raise ValueError('Not all keys matched the expression')
        groups = [m.group(1) for m in matches if m is not None]
        keys_to_split, inverse_idx, counts = np.unique(groups,
                                                       return_inverse=True,
                                                       return_counts=True)
        return keys_to_split, inverse_idx, counts

    @staticmethod
    def __balanced_sep(split: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Returns the most balanced separation"""
        counts_cum = np.cumsum(counts)
        sep = np.cumsum(split) * counts_cum[-1]
        res = np.argmin(np.abs(counts_cum[:, np.newaxis] - sep), axis=0) + 1
        return res

    @staticmethod
    def __split_keys(keys: np.ndarray, split: np.ndarray,
                     pattern: str) -> List[np.ndarray]:
        to_split, inverse_idx, counts = Splitter.__get_groups(keys, pattern)
        shuffled_gid = np.arange(0, len(to_split))
        np.random.shuffle(shuffled_gid)
        shuffled_counts = counts[shuffled_gid]

        split_keys = []
        c = 0
        for s in Splitter.__balanced_sep(split, shuffled_counts):
            idx = [k in shuffled_gid[c:s] for k in inverse_idx]
            split_keys.append(keys[idx])
            c = s
        return split_keys
