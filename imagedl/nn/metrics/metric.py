"""Base metric with some additional methods"""
from abc import abstractmethod
from typing import Tuple, Any, List

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from torch import Tensor

from imagedl.nn.metrics.instance import MeanMetric


class UpgradedMetric(Metric):
    """Metric with protected method for aggregating results and mean method
    To instantiate class you need to override following methods:
    - _reset - reset your variables (default - pass)
    - _update - update your variables (default - pass)
    - compute - compute metric
    - visualize - visualize your metric in graph if possible (default - pass)
    """

    def __init__(self, output_transform=lambda x: x, vis: bool = False):
        super().__init__(output_transform)
        self._apply_reset = True
        self._updates = 0
        self._vis = vis

    @reinit__is_reduced
    def reset(self) -> None:
        """Resets the metric"""
        self._apply_reset = True

    def _reset(self) -> None:
        pass

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        pass

    @abstractmethod
    def compute(self) -> Any:
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

    def visualize(self, value: torch.Tensor, legend: List[str] = None) -> None:
        """Visualize metrics info"""
        raise NotImplementedError()

    @property
    def vis(self):
        return self._vis
