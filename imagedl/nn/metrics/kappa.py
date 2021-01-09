"""Quadratic Kappa"""
from typing import Tuple

import torch

from imagedl.data.datasets.abstract import Transform
from .metric import UpgradedMetric


class QuadraticKappa(UpgradedMetric):
    """Kappa with quadratic weights"""

    def __init__(self, n_classes: int = 1,
                 output_transform: Transform = lambda x: x):
        r = torch.arange(0, n_classes, dtype=torch.float)
        self._weight_matrix = torch.stack([r] * n_classes)
        self._weight_matrix -= torch.stack([r] * n_classes, 1)
        self._weight_matrix **= 2
        self._weight_matrix /= (n_classes - 1) ** 2
        self._histogram = torch.zeros(n_classes, n_classes)
        self._n_classes = n_classes
        super().__init__(output_transform=output_transform)

    def _reset(self) -> None:
        """Resets the metric"""
        self._histogram = torch.zeros(self._n_classes, self._n_classes)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = output
        self._histogram = self._histogram.to(logits.device)
        self._weight_matrix = self._weight_matrix.to(logits.device)

        pred = logits.argmax(1)
        stacked = torch.stack([pred, targets], 1).to(logits.device)
        uq, cnt = stacked.unique(dim=0, return_counts=True)
        self._histogram[uq[:, 0], uq[:, 1]] += cnt

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        assert self._histogram.sum() > 0
        random = self._histogram.sum(1)[:, None] @ self._histogram.sum(0)[None]
        random /= (self._histogram.sum() ** 2 + 1e-5)
        histogram = self._histogram / (self._histogram.sum() + 1e-5)
        hst = (histogram * self._weight_matrix).sum()
        rnd = (random * self._weight_matrix).sum()
        return 1 - hst / (rnd + 1e-7)
