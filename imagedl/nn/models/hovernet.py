"""HoverNet"""
from typing import Tuple

from torch import nn, Tensor

from .blocks import Encoder, Decoder, SegmentationHead


class HoverNet(nn.Module):
    """HoverNet"""

    def __init__(self, n_classes: int = 1, increased=False, remove_hv=False,
                 add_np=False, n_dense: Tuple[int, int] = (1, 2), bias=False):
        super(HoverNet, self).__init__()
        self.n_classes = n_classes
        self.remove_hv = remove_hv
        self.add_np = add_np

        self.encoder = Encoder(increased, bias=bias)
        if self.n_classes > 1:
            self.decoder_nc = Decoder(1024, n_dense[0], n_dense[1], bias=bias)
            self.head_nc = SegmentationHead(n_channels=n_classes)

        if add_np:
            self.decoder_np = Decoder(1024, n_dense[0], n_dense[1], bias=bias)
            self.head_np = SegmentationHead()

        if not remove_hv:
            self.decoder_hv = Decoder(1024, n_dense[0], n_dense[1], bias=bias)
            self.head_hv = SegmentationHead(n_channels=2)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        """Forward"""
        x = self.encoder(inputs)
        out_list = []
        if self.add_np:
            out_list.append(self.head_np(self.decoder_np(x)))
        if not self.remove_hv:
            out_list.append(self.head_hv(self.decoder_hv(x)))
        if self.n_classes > 1:
            out_list.append(self.head_nc(self.decoder_nc(x)))
        return tuple(out_list)
