import os
import tempfile

import matplotlib.pyplot as plt
import torch
from ignite.contrib.handlers import TensorboardLogger
from skimage import io


class PlotSave:

    def __init__(self, tag: str, tb_logger: TensorboardLogger, epoch: int):
        self.tb_logger = tb_logger
        self.tag = tag
        self.epoch = epoch
        self.filename = tempfile.NamedTemporaryFile(suffix='.png').name

    def __enter__(self):
        plt.figure()

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.savefig(self.filename, format='png')
        image = io.imread(self.filename)
        image = torch.tensor(image).permute(2, 0, 1)
        self.tb_logger.writer.add_image(self.tag, image, self.epoch)
        os.remove(self.filename)
