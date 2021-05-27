"""Mish activation"""
import torch
import torch.nn.functional as f
from torch import nn


class Mish(nn.Module):
    """Mish Activation block"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate Mish activation"""
        return inputs * torch.tanh(f.softplus(inputs, inplace=self.inplace))
