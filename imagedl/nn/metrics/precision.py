"""Precision metric"""
from typing import Tuple

import torch

from imagedl.data.datasets.abstract import Transform
from imagedl.nn.metrics.metric import UpgradedMetric, sum_class_agg


class Precision(UpgradedMetric):
    """Precision metric. Input: logits, targets"""

    def __init__(self, n_classes: int = 1,
                 output_transform: Transform = lambda x: x, multi_label=False):
        super().__init__(output_transform=output_transform)
        self._tp = torch.zeros(n_classes)
        self._p = torch.zeros(n_classes)
        self._n_classes = n_classes
        self._multi_label = multi_label

    def _reset(self) -> None:
        """Resets the metric"""
        self._tp = torch.zeros(self._n_classes)
        self._p = torch.zeros(self._n_classes)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = output
        self._tp = self._tp.to(logits.device)
        self._p = self._p.to(logits.device)
        device = logits.device

        if not self._multi_label:
            if self._n_classes > 1:
                pred = logits.argmax(1)
                tp: torch.Tensor = pred == targets
                values = torch.ones(int(tp.sum()), device=device)
                self._tp += sum_class_agg(targets[tp], values, self._n_classes)
                uq, cnt = pred.unique(return_counts=True)
                self._p[uq] += cnt
            else:
                pred = 1 * (logits > 0)
                tp: torch.Tensor = pred * targets
                self._tp[0] += tp.sum()
                self._p[0] += pred.sum()

        else:
            pred = logits > 0
            tp = (pred * targets).sum(0)
            self._tp += tp
            self._p += pred.sum(0)

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        return self._tp / (self._p + 1e-7)
