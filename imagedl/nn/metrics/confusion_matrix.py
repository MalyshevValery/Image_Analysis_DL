"""Confusion matrix metric"""
from typing import Tuple, List

import torch
from ignite.metrics.metric import sync_all_reduce
from seaborn import heatmap

from imagedl.data.datasets.abstract import Transform
from imagedl.nn.metrics.metric import UpgradedMetric


class ConfusionMatrix(UpgradedMetric):
    """
    Confusion matrix with quantifying
    
    :param n_classes: Number of classes (1 for binary case)
    :param output_transform: Transform for ignite.engine output before applying
        this metric
    """

    def __init__(self, n_classes: int = 1,
                 output_transform: Transform = lambda x: x):
        super().__init__(output_transform, True)
        self._is_binary = 1 if n_classes == 1 else 0
        self._n_classes = n_classes + self._is_binary
        self._matrix = torch.zeros(self._n_classes, self._n_classes)

    def _reset(self) -> None:
        self._matrix *= 0.0

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = self._prepare(*output)
        if self._matrix.device != logits.device:
            self._matrix = self._matrix.to(logits.device)
        if self._is_binary == 1:
            probs = 1 * (logits > 0)
        else:
            probs = torch.argmax(logits, dim=1).long()
            targets = targets[:, 0]
        stacked = torch.stack([probs, targets]).view(2, -1)
        pairs, counts = stacked.unique(dim=1, return_counts=True)
        matrix = torch.zeros(self._n_classes, self._n_classes,
                             device=logits.device)
        matrix[pairs[0], pairs[1]] = counts.float()

        t_un, t_cnt = targets.unique(return_counts=True)
        matrix[:, t_un] /= t_cnt
        self._matrix += matrix

    @sync_all_reduce('_updates', '_matrix')
    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        assert self._updates > 0
        return self._matrix / (self._matrix.sum() + 1e-7)

    @property
    def n_classes(self) -> int:
        """Return number of classes"""
        return self._n_classes - self._is_binary

    def _prepare(self, logits: torch.Tensor,
                 targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(logits.shape) - len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        targets = targets.long()
        assert logits.shape[1] == self._n_classes - self._is_binary
        return logits, targets

    def visualize(self, value: torch.Tensor, legend: List[str] = None) -> None:
        """Visualizing method for Confusion Matrix

        :param value: Tensor value of metric to plot
        :param legend: Optional legend from config
        """
        heatmap(value.cpu().numpy(), annot=True, xticklabels=legend,
                yticklabels=legend)
