"""Combined Loss"""
from typing import Iterable

import torch
from torch import nn


class CombinedLoss(nn.Module):
    """Combination of losses"""

    def __init__(self, *args: nn.Module, weights: Iterable[float] = None):
        self.losses = args
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0] * len(self.losses)
        super().__init__()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Calculate losses"""
        iterable = zip(self.weights, self.losses)
        loss_values = [w * l(logits, targets) for w, l in iterable]
        return torch.stack(loss_values).sum()
