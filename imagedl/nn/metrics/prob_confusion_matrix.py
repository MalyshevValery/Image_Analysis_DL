"""Confusion Matrix with probabilities"""
from typing import Tuple

import torch

from .confusion_matrix import ConfusionMatrix


class ProbConfusionMatrix(ConfusionMatrix):
    """
    Confusion Matrix which works with probabilities of classes instead of
    classes
    """

    def _prepare(self, logits: torch.Tensor,
                 targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = targets.long()
        if self._is_binary == 1 or self._multi_label:
            probs = logits.sigmoid()
            probs = torch.stack([1 - probs, probs], -1)
        else:
            probs = torch.softmax(logits, dim=-1)
        return probs, targets
