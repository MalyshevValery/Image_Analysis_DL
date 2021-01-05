import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))