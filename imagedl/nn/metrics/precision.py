"""Precision metric"""
from typing import Tuple

import torch

from imagedl.data.datasets.abstract import Transform
from imagedl.nn.metrics.metric import UpgradedMetric, sum_class_agg


class Precision(UpgradedMetric):
    """Precision metric. Input: logits, targets"""
    def __init__(self, n_classes: int = 1,
                 output_transform: Transform = lambda x: x):
        super().__init__(output_transform=output_transform)
        self._scores = torch.zeros(n_classes)
        self._n_classes = n_classes

    def _reset(self) -> None:
        """Resets the metric"""
        self._scores = torch.zeros(self._n_classes)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = output
        self._scores = self._scores.to(logits.device)
        device = logits.device

        pred = logits.argmax(1)
        tp: torch.Tensor = pred == targets
        values = torch.ones(int(tp.sum()), device=device)
        tp = sum_class_agg(targets[tp], values, self._n_classes)
        uq, cnt = pred.unique(return_counts=True)
        self._scores[uq] += tp[uq] / cnt

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        return self._scores / (self._updates + 1e-7)
