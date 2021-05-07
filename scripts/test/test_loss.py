"""Double binary loss"""
from torch import nn, Tensor


class TestLoss(nn.Module):
    """Double BCE for test script"""

    def __init__(self) -> None:
        super().__init__()
        self.loss1 = nn.BCEWithLogitsLoss()
        self.loss2 = nn.BCEWithLogitsLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Calculate loss"""
        return self.loss1(pred[0], target[0]) + self.loss2(pred[1], target[1])
