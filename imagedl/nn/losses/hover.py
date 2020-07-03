"""Loss for HoverNet"""
from typing import Tuple

import torch
from torch import nn

from .combined import CombinedLoss
from .dice import DiceLoss


class HoverLoss(nn.Module):
    """Loss for HoverNet"""

    def __init__(self, classification: bool = False):
        super(HoverLoss, self).__init__()
        self.losses = {
            'NC_E': nn.BCEWithLogitsLoss() if not classification else nn.CrossEntropyLoss(),
            'NC_Dice': DiceLoss(),
            'HV_MSE': nn.MSELoss(),
        }
        self.nc_loss = CombinedLoss(self.losses['NC_E'],
                                    self.losses['NC_Dice'])
        self.hv_loss = self.losses['HV_MSE']

    def forward(self, logits: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Calculate loss"""
        loss = self.nc_loss(logits[0], targets[0])
        loss += self.hv_loss(logits[1], targets[1])
        return loss
