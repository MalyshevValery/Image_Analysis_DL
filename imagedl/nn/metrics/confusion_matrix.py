"""Confusion matrix metric"""
from typing import Tuple, List

import torch
from seaborn import heatmap

from imagedl.nn.metrics.metric import UpgradedMetric
from imagedl.utils.types import MetricTransform


class ConfusionMatrix(UpgradedMetric):
    """
    Confusion matrix with quantifying
    
    :param n_classes: Number of classes (1 for binary case)
    :param multi_label: If classes are independent
    :param output_transform: Transform for ignite.engine output before applying
        this metric
    """

    def __init__(self, n_classes: int = 1, multi_label: bool = False,
                 output_transform: MetricTransform = lambda x: x):
        super().__init__(output_transform, True)
        self._is_binary = 1 if n_classes == 1 else 0
        self._multi_label = multi_label
        self._n_classes = n_classes + self._is_binary
        if self._multi_label:
            self._matrix = torch.zeros(2, 2 * self._n_classes)
        else:
            self._matrix = torch.zeros(self._n_classes, self._n_classes)

    def _reset(self) -> None:
        self._matrix *= 0.0

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates the metric"""
        probs, targets = self._prepare(*output)
        if self._matrix.device != probs.device:
            self._matrix = self._matrix.to(probs.device)
        new_targets = torch.zeros(probs.shape, device=probs.device)
        targets = new_targets.scatter_(-1, targets[..., None], 1.0)
        if self._multi_label:
            probs = probs.reshape(probs.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
            all_classes = torch.matmul(probs.T, targets)
            for i in range(0, self._n_classes * 2, 2):
                self._matrix[:, i:i + 2] += all_classes[i:i + 2, i:i + 2]
        else:
            self._matrix += torch.matmul(probs.T, targets)

    def compute(self) -> torch.Tensor:
        """Metric aggregation"""
        assert self._updates > 0
        return self._matrix / (self._matrix.sum(0) + 1e-7)

    @property
    def n_classes(self) -> int:
        """Return number of classes"""
        return self._n_classes - self._is_binary

    def _prepare(self, logits: torch.Tensor,
                 targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = targets.long()
        if self._is_binary == 1 or self._multi_label:
            probs = (logits > 0) * 1.0
            probs = torch.stack([-probs + 1, probs], -1)
        else:
            new_probs = torch.zeros(logits.shape, device=logits.device)
            prob_idx = torch.argmax(logits, -1)[..., None]
            probs = new_probs.scatter_(-1, prob_idx, 1.0)
        return probs, targets

    def visualize(self, value: torch.Tensor, legend: List[str] = None) -> None:
        """Visualizing method for Confusion Matrix

        :param value: Tensor value of metric to plot
        :param legend: Optional legend from config
        """
        heatmap(value.cpu().numpy(), annot=True, xticklabels=legend,
                yticklabels=legend)
