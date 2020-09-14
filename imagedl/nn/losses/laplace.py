"""Quantile Loss"""
import math

import numpy as np
import torch
from torch import nn


class LaplaceLoss(nn.Module):
    """Laplace Loss"""

    def __init__(self, get_mean_std, clips=None):
        super().__init__()
        self.get_mean_std = get_mean_std
        self.root_2 = math.sqrt(2)
        self.clips = clips

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        means, std = self.get_mean_std(logits)
        delta = torch.abs(means - targets)
        if self.clips is not None:
            delta[delta > self.clips[0]] = self.clips[0]
            std[std < self.clips[1]] = self.clips[1]
        laplace = np.sqrt(2) * delta / std + torch.log(np.sqrt(2) * std)
        return laplace.mean()
