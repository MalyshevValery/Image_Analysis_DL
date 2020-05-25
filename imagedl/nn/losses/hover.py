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
            'NP_BCE': nn.BCEWithLogitsLoss(),
            'NP_Dice': DiceLoss(),
            'HV_MSE': nn.MSELoss(),
        }
        self.np_loss = CombinedLoss(self.losses['NP_BCE'],
                                    self.losses['NP_Dice'])
        self.hv_loss = self.losses['HV_MSE']
        self.classification = classification
        if classification:
            self.losses['NC_CE'] = nn.CrossEntropyLoss()
            self.losses['NC_Dice'] = DiceLoss()
            self.nc_loss = CombinedLoss(self.losses['NC_CE'],
                                        self.losses['NC_Dice'])

    def forward(self, logits: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Calculate loss"""
        loss = self.np_loss(logits[0], targets[0])
        loss += self.hv_loss(logits[1], targets[1])
        if self.classification:
            loss += self.nc_loss(logits[2], targets[2])
        if torch.isnan(loss):
            print('NAAAAAN')
        return loss
