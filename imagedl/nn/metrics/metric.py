"""Base metric with some additional methods"""
from abc import abstractmethod
from typing import Tuple, Any, List

import torch
from ignite.metrics import Metric
from torch import Tensor

from imagedl.utils.types import MetricTransform


class MeanMetric(Metric):
    """Metric which computes mean of source metric"""

    def __init__(self, source: Metric) -> None:
        self._source = source
        super().__init__()

    def reset(self) -> None:
        """Nothing to reset"""
        pass

    def update(self, output: Any) -> None:
        """Nothing to update"""
        pass

    def compute(self) -> float:
        """Computes mean of parent metric"""
        return float(self._source.compute().mean().item())


class UpgradedMetric(Metric):
    """Metric with protected method for aggregating results and mean method
    To instantiate class you need to override following methods:
    - _reset - reset your variables (default - pass)
    - _update - update your variables (default - pass)
    - compute - compute metric
    - visualize - visualize your metric in graph if possible (default - pass)
    """

    def __init__(self,
                 output_transform: MetricTransform = lambda x: x,
                 vis: bool = False) -> None:
        super().__init__(output_transform)
        self._apply_reset = True
        self._updates = 0
        self._vis = vis

    def reset(self) -> None:
        """Resets the metric"""
        self._apply_reset = True

    def _reset(self) -> None:
        pass

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Compute metric final values"""
        raise NotImplementedError()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Update metric internal state"""
        if self._apply_reset:
            self._reset()
            self._apply_reset = False
            self._updates = 1
        else:
            self._updates += 1
        self._update(output)

    def mean(self) -> MeanMetric:
        """Returns metric which takes mean if this one returns array"""
        return MeanMetric(self)

    def visualize(self, value: torch.Tensor, legend: List[str] = None) -> None:
        """Visualize metrics info"""
        raise Exception("Not available for current metric")

    @property
    def vis(self) -> bool:
        """Getter for visualization flag"""
        return self._vis


def sum_class_agg(labels: Tensor, values: Tensor,
                  n_classes: int = 1) -> Tensor:
    """Sums values according to given labels

    :param labels: Labels with indexes for values
    :param values: Values to sum
    :param n_classes: number of classes
    :return:
    """
    res = torch.zeros(n_classes, device=values.device)
    res.scatter_add_(0, labels.long(), values)
    return res
