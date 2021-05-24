"""Saver for matplotlib plots"""
import os
import tempfile
from types import TracebackType
from typing import Optional, Type

import matplotlib.pyplot as plt
import torch
from ignite.contrib.handlers import TensorboardLogger
from skimage import io


class PlotSave:
    """Context manager for saving matplotlib plots"""

    def __init__(self, tag: str, tb_logger: TensorboardLogger, epoch: int):
        self.tb_logger = tb_logger
        self.tag = tag
        self.epoch = epoch
        self.filename = tempfile.NamedTemporaryFile(suffix='.png').name
        self.fig = None

    def __enter__(self) -> None:
        self.fig = plt.figure()

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        plt.savefig(self.filename, format='png')
        image = io.imread(self.filename)
        image = torch.tensor(image).permute(2, 0, 1)
        self.tb_logger.writer.add_image(self.tag, image, self.epoch)
        os.remove(self.filename)
        plt.close(self.fig)
