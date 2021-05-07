"""Model for testing with two inputs and outputs"""
from typing import Tuple

import torch
from torch import nn, Tensor


class TestModel(nn.Module):
    """TestModel, takes two vectors and perform two binary classifications"""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.enc1 = TestModel.__get_encoder(n)
        self.enc2 = TestModel.__get_encoder(n)
        self.dec1 = TestModel.__get_decoder()
        self.dec2 = TestModel.__get_decoder()

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward method"""
        x1, x2 = x
        f1 = self.enc1(x1)
        f2 = self.enc2(x2)
        f = torch.cat((f1, f2), 1)
        log1 = self.dec1(f)
        log2 = self.dec2(f)
        return log1, log2

    @staticmethod
    def __block(inp: int, out: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(inp, out),
            nn.BatchNorm1d(out),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

    @staticmethod
    def __get_encoder(in_feat: int) -> nn.Module:
        return nn.Sequential(
            TestModel.__block(in_feat, 100),
            TestModel.__block(100, 256),
            TestModel.__block(256, 128),
            TestModel.__block(128, 128),
            TestModel.__block(128, 64),
            TestModel.__block(64, 64),
        )

    @staticmethod
    def __get_decoder():
        return nn.Sequential(
            TestModel.__block(128, 128),
            TestModel.__block(128, 128),
            TestModel.__block(128, 64),
            TestModel.__block(64, 32),
            nn.Linear(32, 1),
            nn.Flatten(0)
        )
