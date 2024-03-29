"""Dice Loss"""
import torch
from torch import nn


class DiceLoss(nn.Module):
    """Dice Loss"""

    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Calculate dice loss"""
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            if len(targets.shape) - len(probs.shape) == 1:
                targets = targets.unsqueeze(1)
        else:
            probs = torch.softmax(logits, dim=1)
            new_targets = torch.zeros(logits.shape, device=probs.device)
            new_targets.scatter_(1, targets.unsqueeze(1), 1.0)
            targets = new_targets

        sum_axis = (0, *range(2, len(probs.shape)))
        intersection = (probs * targets).sum(dim=sum_axis)
        union = probs.sum(dim=sum_axis) + targets.sum(dim=sum_axis)
        dice_score = torch.mean(2 * intersection / (union + self.smooth))
        return - (dice_score + 1e-7).log()
