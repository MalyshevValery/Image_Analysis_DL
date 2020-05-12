"""HoverNet"""
from typing import Tuple

from torch import nn, Tensor

from .blocks import Encoder, Decoder, SegmentationHead


class HoverNet(nn.Module):
    """HoverNet"""

    def __init__(self, n_classes: int = 1):
        super(HoverNet, self).__init__()
        self.n_classes = n_classes

        self.encoder = Encoder()
        self.decoder_np = Decoder(1024)
        self.decoder_hv = Decoder(1024)
        self.head_np = SegmentationHead()
        self.head_hv = SegmentationHead(n_channels=2)
        if n_classes > 1:
            self.decoder_nc = Decoder(1024)
            self.head_nc = SegmentationHead(n_channels=n_classes)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        """Forward"""
        x = self.encoder(inputs)
        out_list = [
            self.head_np(self.decoder_np(x)),
            self.head_hv(self.decoder_hv(x)),
        ]
        if self.n_classes > 1:
            out_list.append(self.head_nc(self.decoder_nc(x)))
        return tuple(out_list)
