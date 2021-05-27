"""Quantile Loss"""
from typing import Sequence

import torch
from torch import nn


class QuantileLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, quantiles: Sequence[float]):
        super().__init__()
        self.qs = torch.tensor(quantiles, dtype=torch.float)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Calculate quantile loss"""
        if self.qs.device != logits.device:
            self.qs = self.qs.to(logits.device)
        delta = targets[..., None] - logits
        first = delta * self.qs[None]
        second = delta * (self.qs[None] - 1)
        res = torch.max(torch.stack([first, second]), 0).values
        return torch.mean(res)
