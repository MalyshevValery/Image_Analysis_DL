"""Negative example miner from static datasets"""
from typing import Tuple, List, Iterator, Sized, Sequence, Optional

import torch
from ignite.metrics import Metric
from torch.utils.data import Sampler
from torch.utils.data.dataset import T_co, Dataset

from imagedl.utils.types import DataType


def _count(data: DataType) -> int:
    if isinstance(data, torch.Tensor):
        return int(data.shape[0])
    elif isinstance(data, Sequence):
        return int(data[0].shape[0])
    else:
        raise ValueError('Could not extract batch size')


def _get(data: DataType, i: int) -> Optional[DataType]:
    if isinstance(data, torch.Tensor):
        return data[i:i + 1]
    elif isinstance(data, Sequence):
        return tuple(d[i:i + 1] for d in data)
    else:
        return None


class ExampleMetric(Metric):
    """Metric which saved"""

    def __init__(self, metric: Metric, metric_list: List[float]):
        self._metric = metric
        self._list = metric_list
        super().__init__()

    def reset(self) -> None:
        """Resets the metric"""
        self._metric.reset()

    def update(self, output: Tuple[DataType, DataType]) -> None:
        """Updates the metric"""

        logits = output[0]
        targets = output[1]

        batch_size = _count(logits)
        assert batch_size == _count(targets)
        for i in range(batch_size):
            self._metric.update((_get(logits, i), _get(targets, i)))
            value = self._metric.compute()
            self._metric.reset()
            self._list.append(value)

    def compute(self) -> torch.Tensor:
        """Computes mean of all calculated values"""
        return torch.tensor(self._list).mean()


class NegExampleSampler(Sampler[int]):
    """Sampler which takes negative examples for learning"""

    def __init__(self, dataset: Dataset[T_co], ratio: float, metric: Metric):
        if not isinstance(dataset, Sized):
            raise TypeError('Dataset provided has no len method')
        super().__init__(dataset)
        self.__dataset: Sized = dataset
        self.__list: List[float] = []
        self.__metric = ExampleMetric(metric, self.__list)

        self.__n = int(len(self.__dataset) * ratio)
        self.__losses = torch.full((len(self.__dataset),), -1.0)
        self.__selected: torch.Tensor
        print(f'Number of samples: {self.__n}')

    def __len__(self) -> int:
        return self.__n

    def __iter__(self) -> Iterator[int]:
        losses = torch.tensor(self.__list)[:self.__n]
        self.__list.clear()
        if len(losses) == len(self.__selected):
            self.__losses[self.__selected] = losses

        skipped = torch.where(self.__losses.eq(-1.0))[0]
        skipped = skipped[torch.randperm(len(skipped))]
        if len(skipped) < self.__n:
            selected_pos = torch.argsort(self.__losses, descending=True)[
                           :(self.__n - len(skipped))]
            self.__selected = torch.cat([skipped, selected_pos])
        else:
            self.__selected = skipped[:self.__n]
        self.__selected = self.__selected[torch.randperm(self.__n)]
        return iter(self.__selected)

    @property
    def metric(self) -> ExampleMetric:
        """Returns metric which is used by sampler
        to define negative examples"""
        return self.__metric
