"""Confusion Matrix with probabilities"""
from typing import Tuple

import torch

from .confusion_matrix import ConfusionMatrix


class ProbConfusionMatrix(ConfusionMatrix):
    """
    Confusion Matrix which works with probabilities of classes instead of
    classes
    """

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Updates this metric"""
        logits, targets = self._prepare(*output)
        if self._is_binary == 1:
            probs = logits.sigmoid()
            probs = torch.cat([1 - probs, probs], dim=1)
            targets = torch.cat([1 - targets, targets], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            new_targets = torch.zeros(probs.shape)
            targets = new_targets.scatter_(1, targets, 1.0)
        probs = probs.permute(1, 0, *range(2, len(probs.shape)))
        targets = targets.permute(1, 0, *range(2, len(targets.shape)))
        probs = torch.stack([probs] * self._n_classes)
        targets = torch.stack([targets] * self._n_classes, 1)
        matrix = (probs * targets).mean(dim=tuple(range(2, len(probs.shape))))
        self._matrix += matrix
