"""Confusion matrix metric"""
from typing import Tuple

import ignite
import torch

from imagedl.data.datasets.abstract import Transform


class ConfusionMatrix(ignite.metrics.Metric):
    """
    Confusion matrix with quantifying
    
    :param n_classes: Number of classes (1 for binary case)
    :param output_transform: Transform for ignite.engine output before applying
        this metric
    """

    def __init__(self, n_classes: int = 1,
                 output_transform: Transform = lambda x: x):
        self._is_binary = 1 if n_classes == 1 else 0
        self._n_classes = n_classes + self._is_binary
        self._updates = 0
        self._apply_reset = False
        self._matrix = torch.zeros(self._n_classes, self._n_classes)
        super().__init__(output_transform=output_transform)

    def reset(self) -> None:
        """Resets the metric"""
        self._apply_reset = True

    def _reset(self) -> None:
        self._matrix *= 0.0
        self._updates = 1

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        logits, targets = self._prepare(*output)
        if self._is_binary == 1:
            probs = 1 * (logits > 0)
        else:
            probs = torch.argmax(logits, dim=1).long()
            targets = targets[:, 0]
        stacked = torch.stack([probs, targets]).view(2, -1)
        pairs, counts = torch.unique(stacked, dim=1, return_counts=True)
        matrix = torch.zeros(self._n_classes, self._n_classes)
        matrix[pairs[0], pairs[1]] = counts.float()
        matrix /= counts.sum()

        if self._apply_reset:
            self._reset()
            self._apply_reset = False
        self._matrix += matrix

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        assert self._updates > 0
        return self._matrix.clone() / self._updates

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
        self._updates += 1
        return logits, targets
