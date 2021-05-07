"""Training algorithm"""
from pathlib import Path

import pandas as pd
import torch

from imagedl.config import Config
from imagedl.data import Split
from .data import get_data_loaders
from .engines import evaluate, evaluator, create_trainer
from .handlers import tensorboard_logger, train_handlers


def train_run(config: Config, split: Split, job_dir: Path,
              device: torch.device, own_split: bool) -> pd.DataFrame:
    """Training procedure depending on split"""
    job_dir.mkdir(parents=True, exist_ok=True)  # Needed in case of KFold
    model, optimizer, criterion, split, trainer = create_trainer(config, device,
                                                                 split,
                                                                 own_split)
    epochs, *_ = config.train
    train_dl, val_dl, test_dl = get_data_loaders(config, split)

    val_evaluator = evaluator(config.test, criterion, model, device)
    tb_logger = tensorboard_logger(trainer, val_evaluator, model, config,
                                   train_dl, val_dl, device)
    progress_bar = train_handlers(config, trainer, val_evaluator, model,
                                  optimizer, split, val_dl, tb_logger)

    trainer.run(train_dl, max_epochs=epochs)

    if len(test_dl) > 0:
        df = evaluate(config, test_dl, split.test, criterion, progress_bar,
                      model, device,
                      tb_logger, trainer)

        return df


def single_training(device: torch.device, config: Config, split: Split,
                    job_dir: Path, own_split: bool) -> pd.DataFrame:
    """Training on a single GPU TODO: Add distributed training"""
    return train_run(config, split, job_dir, device, own_split)
