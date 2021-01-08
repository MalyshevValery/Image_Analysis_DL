from typing import Tuple

import torch
from ignite.metrics.metric import sync_all_reduce

from imagedl.data.datasets.abstract import Transform
from imagedl.nn.metrics.metric import UpgradedMetric


class Recall(UpgradedMetric):
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
        tp = pred == targets
        tp = self._sum_class_agg(targets[tp], torch.ones(tp.sum(),
                                                         device=device),
                                 self._n_classes)
        uq, cnt = targets.unique(return_counts=True)
        self._scores[uq] += tp[uq] / cnt

    @sync_all_reduce('_updates', '_scores')
    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        return self._scores / (self._updates + 1e-7)
