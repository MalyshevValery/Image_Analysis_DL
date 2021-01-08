"""Base metric with some additional methods"""
from abc import abstractmethod
from typing import Tuple

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from torch import Tensor

from imagedl.nn.metrics.instance import MeanMetric


class UpgradedMetric(Metric):
    """Metric with protected method for aggregating results and mean method"""

    def __init__(self, output_transform=lambda x: x):
        super().__init__(output_transform)
        self._apply_reset = True
        self._updates = 0

    @reinit__is_reduced
    def reset(self) -> None:
        """Resets the metric"""
        self._apply_reset = True

    @abstractmethod
    def _reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        if self._apply_reset:
            self._reset()
            self._apply_reset = False
            self._updates = 1
        else:
            self._updates += 1
        self._update(output)

    def _sum_class_agg(self, labels: Tensor, values: Tensor,
                       n_classes: int) -> Tensor:
        res = torch.zeros(n_classes if n_classes is not None else 1,
                          device=values.device)
        res.scatter_add_(0, labels, values)
        return res

    def mean(self) -> MeanMetric:
        """Returns metric which takes mean if this one returns array"""
        return MeanMetric(self)
