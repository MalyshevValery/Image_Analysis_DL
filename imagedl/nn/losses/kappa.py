"""Kappa Loss"""
import torch
from torch import nn


class KappaLoss(nn.Module):
    """Kappa Loss"""

    def __init__(self, n_classes: int):
        super().__init__()
        r = torch.arange(0, n_classes, dtype=torch.float)
        self._weight_matrix = torch.stack([r] * n_classes, 0) - torch.stack([r] * n_classes, 1)
        self._weight_matrix = self._weight_matrix ** 2
        self._weight_matrix /= (n_classes - 1) ** 2
        self._n_classes = n_classes
        self.smooth = 1e-5

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss"""
        self._weight_matrix = self._weight_matrix.to(logits.device)
        targets_prob = torch.zeros(logits.shape, device=logits.device)
        targets_prob[range(logits.shape[0]), targets] = 1.0
        logits_prob = logits.softmax(1)
        hist = logits_prob.transpose(0, 1) @ targets_prob
        random = logits_prob.sum(0)[:, None] @ targets_prob.sum(0)[None]
        random /= (hist.sum() ** 2 + self.smooth)
        random = random * self._weight_matrix

        hist /= (hist.sum() + self.smooth)
        hist = hist * self._weight_matrix
        k = hist.sum() / (random.sum() + self.smooth)
        return k
