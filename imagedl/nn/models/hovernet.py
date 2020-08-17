"""HoverNet"""
from typing import Tuple

from torch import nn, Tensor

from .blocks import Encoder, Decoder, SegmentationHead


class HoverNet(nn.Module):
    """HoverNet"""

    def __init__(self, n_classes: int = 1, increased=False, remove_hv=False,
                 n_dense: Tuple[int, int] = (1, 2)):
        super(HoverNet, self).__init__()
        self.n_classes = n_classes
        self.remove_hv = remove_hv

        self.encoder = Encoder(increased)
        self.decoder_nc = Decoder(1024, n_dense[0], n_dense[1])
        self.head_nc = SegmentationHead(n_channels=n_classes)

        if not remove_hv:
            self.decoder_hv = Decoder(1024, n_dense[0], n_dense[1])
            self.head_hv = SegmentationHead(n_channels=2)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        """Forward"""
        x = self.encoder(inputs)
        if self.remove_hv:
            out_list = [
                self.head_nc(self.decoder_nc(x)),
                self.head_hv(self.decoder_hv(x)),
            ]
            return tuple(out_list)
        else:
            return self.head_nc(self.decoder_nc(x))
