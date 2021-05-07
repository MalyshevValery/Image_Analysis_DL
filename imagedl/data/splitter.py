"""Classes for data splitting"""
from typing import NamedTuple, Tuple, List, Generator, Mapping

import numpy as np
import torch

_TrainValTest = Tuple[float, float, float]


class Split(NamedTuple):
    """Tuple with train, validation and test DataGenerators"""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def state_dict(self):
        return {
            'train': torch.tensor(self.train),
            'val': torch.tensor(self.val),
            'test': torch.tensor(self.test),
        }

    @staticmethod
    def load_state_dict(state_dict: Mapping):
        return Split(
            train=state_dict['train'].numpy(),
            val=state_dict['val'].numpy(),
            test=state_dict['test'].numpy()
        )


class Splitter:
    """Class which splits data into train validation and test sets

    :param total: Total number of entries
    :param group_labels: This argument is used to divide indexes into some
        groups and during splits one group won't be in two sets simultaneously
    """

    def __init__(self, total: int, group_labels: np.ndarray = None):
        self.__total = total
        np.random.seed(42)
        if group_labels is not None:
            self.__group_labels = group_labels
            if self.__total != len(group_labels):
                raise ValueError('Total is not equal to group labels lengths')
        else:
            self.__group_labels = np.arange(self.__total, dtype=np.int32)

    def random_split(self, train_val_test: _TrainValTest) -> Split:
        """Splits indices randomly on three groups to create training, test and
                validation sets.

        :param train_val_test: tuple or list of three elements with sum of 1,
            which contains fractures of whole set for train/validation/test
            split
        :returns Shuffled keys for train , validation, test split
        """
        if sum(train_val_test) > 1:
            raise ValueError('Split', train_val_test, 'is greater than 1')
        if len(train_val_test) != 3:
            raise ValueError('Split', train_val_test, 'must have 3 elements')

        indexes = np.arange(self.__total)
        np.random.shuffle(indexes)
        split_keys = self.__split(indexes, self.__group_labels,
                                  np.array(train_val_test))
        return Split(*split_keys)

    def k_fold(self, val: float, n_fold: int) -> Generator[Split, None, None]:
        """Returns n_fold splits according to K-Fold technique on shuffled keys

        :param val: Fracture of validation part in train fold
        :param n_fold: Number of folds
        """
        if not 0 < val < 1:
            raise ValueError(f'Validation frac {val} must be in (0,1) interval')
        if not 2 <= n_fold <= self.__total:
            raise ValueError(f'Number of folds must be in [2, len(keys)]')

        indexes = np.arange(self.__total)
        np.random.shuffle(indexes)
        split = np.full(n_fold, 1 / n_fold)
        tv_split = np.array([1 - val, val])
        split_indexes = self.__split(indexes, self.__group_labels, split)

        for i in range(n_fold):
            test_indexes = split_indexes[i]
            tv_indexes = [k for j, k in enumerate(split_indexes) if j != i]
            tv_indexes = np.concatenate(tv_indexes)
            t_v_keys = self.__split(tv_indexes, self.__group_labels[tv_indexes],
                                    tv_split)
            yield Split(t_v_keys[0], t_v_keys[1], test_indexes)

    @staticmethod
    def __balanced_sep(split: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Returns the most balanced separation"""
        counts_cum = np.cumsum(counts)
        sep = np.cumsum(split) * counts_cum[-1]
        res = np.argmin(np.abs(counts_cum[:, np.newaxis] - sep), axis=0) + 1
        return res

    @staticmethod
    def __split(indexes: np.ndarray, groups: np.ndarray,
                split: np.ndarray) -> List[np.ndarray]:
        to_split, inverse_idx, counts = np.unique(groups, return_inverse=True,
                                                  return_counts=True)
        shuffled_gid = np.arange(0, len(to_split))
        np.random.shuffle(shuffled_gid)
        shuffled_counts = counts[shuffled_gid]

        split_keys = []
        c = 0
        for s in Splitter.__balanced_sep(split, shuffled_counts):
            idx = [k in shuffled_gid[c:s] for k in inverse_idx]
            split_keys.append(indexes[idx])
            c = s
        return split_keys
