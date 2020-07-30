from typing import Tuple, List

import torch
from ignite.metrics import Metric
from torch.utils.data import Sampler


class ExampleMetric(Metric):
    def __init__(self, metric: Metric, metric_list: List[float]):
        self._metric = metric
        self._list = metric_list
        super().__init__()

    def reset(self) -> None:
        """Resets the metric"""
        self._metric.reset()

    def update(self, output) -> None:
        """Updates the metric"""

        logits = output[0]
        targets = output[1]

        batch_size = self.__count(logits)
        assert batch_size == self.__count(targets)
        for i in range(batch_size):
            self._metric.update((self.__get(logits, i), self.__get(targets, i)))
            value = self._metric.compute()
            self._metric.reset()
            self._list.append(value)

    def compute(self) -> torch.Tensor:
        return torch.tensor(self._list).mean()

    def __count(self, data):
        if isinstance(data, torch.Tensor):
            return data.shape[0]
        elif isinstance(data, Tuple) or isinstance(data, List):
            return data[0].shape[0]
        else:
            return None

    def __get(self, data, i):
        if isinstance(data, torch.Tensor):
            return data[i:i + 1]
        elif isinstance(data, Tuple) or isinstance(data, List):
            return tuple(d[i:i + 1] for d in data)
        else:
            return None


class NegExampleSampler(Sampler):
    def __init__(self, dataset, ratio, metric):
        self.__dataset = dataset
        self.__list = []
        self.__metric = ExampleMetric(metric, self.__list)
        self.__n = int(len(self.__dataset) * ratio)
        self.__losses = torch.full((len(self.__dataset),), -1.0)
        self.__selected = []
        print(f'Number of samples: {self.__n}')

    def __len__(self):
        return self.__n

    def __iter__(self):
        losses = torch.tensor(self.__list)[:self.__n]
        self.__list.clear()
        if len(losses) == len(self.__selected):
            self.__losses[self.__selected] = losses

        skipped = torch.where(self.__losses == -1.0)[0]
        skipped = skipped[torch.randperm(len(skipped))]
        if len(skipped) < self.__n:
            selected_pos = torch.argsort(self.__losses, descending=True)[:(self.__n - len(skipped))]
            self.__selected = torch.cat([skipped, selected_pos])
        else:
            self.__selected = skipped[:self.__n]
        self.__selected = self.__selected[torch.randperm(self.__n)]
        print(self.__selected, self.__losses)
        return iter(self.__selected)

    @property
    def metric(self):
        return self.__metric
