from functools import wraps
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from imagedl.config import Config
from imagedl.data import Split
from .data import get_data_loaders
from .engines import evaluate, evaluator, create_trainer
from .handlers import tensorboard_logger, train_handlers


def train_run(config: Config, split: Split, job_dir: Path,
              device) -> pd.DataFrame:
    """Training procedure depending on split"""
    job_dir.mkdir(parents=True, exist_ok=True)
    model, optimizer, criterion, split, trainer = create_trainer(config, device,
                                                                 split)
    model, optimizer_fn, criterion, checkpoint = config.model_config
    epochs, batch_size, patience = config.train
    train_dl, val_dl, test_dl = get_data_loaders(config, split)

    val_evaluator = evaluator(config.test, criterion, model, device)
    progress_bar = train_handlers(config, trainer, val_evaluator, model,
                                  optimizer, split, val_dl)
    tb_logger = tensorboard_logger(trainer, val_evaluator, model, config,
                                   train_dl, val_dl, device)

    trainer.run(train_dl, max_epochs=epochs)
    tb_logger.close()

    df = evaluate(config, test_dl, split.test, progress_bar, model, device)

    return df


def dist_training(distributed, config: Config, split: Split,
                  job_dir: Path):
    @wraps(train_run)
    def process_run(rank):
        print(f"Running Distributed on {rank}:{distributed[rank]}.")
        world_size = len(distributed)
        device = distributed[rank]
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", init_method='file:///tmp/nccl.dat',
                                world_size=world_size, rank=rank)
        train_run(config, split, job_dir, device)
        dist.destroy_process_group()

    mp.spawn(process_run, nprocs=len(distributed))


def single_training(device, config: Config, split: Split,
                    job_dir: Path):
    return train_run(config, split, job_dir, device)
