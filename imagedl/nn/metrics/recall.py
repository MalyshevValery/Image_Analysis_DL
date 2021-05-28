"""Recall metric"""
from typing import Tuple

import torch

from imagedl.nn.metrics.metric import UpgradedMetric, sum_class_agg
from imagedl.utils.types import MetricTransform


class Recall(UpgradedMetric):
    """Recall metric. Input: logits, targets
    Shapes for binary:
    - Logits [BatchSize]
    - Targets {0,1} [BatchSize]
    Shapes for multi-label:
    - Logits [BatchSize, NClass]
    - Targets {0,1} [BatchSize, NClass]
    Shapes for classification:
    - Logits [BatchSize, NClass]
    - Targets {0,NClass-1} [BatchSize]
    """

    def __init__(self, n_classes: int = 1,
                 output_transform: MetricTransform = lambda x: x,
                 multi_label: bool = False):
        super().__init__(output_transform=output_transform)
        self._tp = torch.zeros(n_classes)
        self._t = torch.zeros(n_classes)
        self._n_classes = n_classes
        self._multi_label = multi_label

    def _reset(self) -> None:
        """Resets the metric"""
        self._tp = torch.zeros(self._n_classes)
        self._t = torch.zeros(self._n_classes)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = output
        self._tp = self._tp.to(logits.device)
        self._t = self._t.to(logits.device)
        device = logits.device

        if not self._multi_label and self._n_classes > 1:
            pred: torch.Tensor = logits.argmax(1)
            tp: torch.Tensor = torch.eq(pred, targets)
            values = torch.ones(int(tp.sum()), device=device)
            self._tp += sum_class_agg(targets[tp], values, self._n_classes)
            uq, cnt = targets.unique(return_counts=True)
            self._t[uq.long()] += cnt
        else:
            pred = torch.gt(logits, 0)
            tp = (pred * targets).sum(0)
            self._tp += tp
            self._t += targets.sum(0)

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        return self._tp / (self._t + 1e-7)
