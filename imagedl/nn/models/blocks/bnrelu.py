"""BatchNorm + ReLU module"""

import torch
from torch import nn


class BNRelu(nn.Module):
    """BatchNorm and ReLU module"""

    def __init__(self, num_features: int):
        super(BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate layer"""
        return self.relu(self.bn(inputs))
