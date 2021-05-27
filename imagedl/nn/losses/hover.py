"""Loss for HoverNet"""
from typing import Tuple, Dict

import torch
from torch import nn

from .combined import CombinedLoss
from .dice import DiceLoss


class HoverLoss(nn.Module):
    """Loss for HoverNet"""

    def __init__(self, np: bool = True, hv: bool = True,
                 classification: bool = True):
        super(HoverLoss, self).__init__()
        self.losses: Dict[str, nn.Module] = {}
        self.__map: Dict[int, nn.Module] = {}
        c = 0
        if np:
            self.losses['NP_E'] = nn.BCEWithLogitsLoss()
            self.losses['NP_Dice'] = DiceLoss()
            self.np_loss = CombinedLoss(self.losses['NP_E'],
                                        self.losses['NP_Dice'])
            self.__map[c] = self.np_loss
            c += 1
        if hv:
            self.losses['HV_MSE'] = nn.MSELoss()
            self.hv_loss = self.losses['HV_MSE']
            self.__map[c] = self.hv_loss
            c += 1
        if classification:
            self.losses['NC_E'] = nn.CrossEntropyLoss()
            self.losses['NC_Dice'] = DiceLoss()
            self.nc_loss = CombinedLoss(self.losses['NC_E'],
                                        self.losses['NC_Dice'])
            self.__map[c] = self.nc_loss
            c += 1
        self.cnt = c

    def forward(self, logits: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Calculate loss"""
        loss = torch.tensor(0)
        for i in range(0, self.cnt):
            loss += self.__map[i](logits[i], targets[i])
        return loss
