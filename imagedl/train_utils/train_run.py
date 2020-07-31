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
    epochs, *_ = config.train
    train_dl, val_dl, test_dl = get_data_loaders(config, split)

    val_evaluator = evaluator(config.test, criterion, model, device)
    tb_logger = tensorboard_logger(trainer, val_evaluator, model, config,
                                   train_dl, val_dl, device)
    progress_bar = train_handlers(config, trainer, val_evaluator, model,
                                  optimizer, split, val_dl, tb_logger)

    trainer.run(train_dl, max_epochs=epochs)

    df = evaluate(config, test_dl, split.test, progress_bar, model, device, tb_logger, trainer)

    return df


def process_run(rank, distributed, config: Config, split: Split, job_dir: Path):
    print(f"Running Distributed on {rank}:{distributed[rank]}.")
    world_size = len(distributed)
    device = distributed[rank]
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method='file:///tmp/nccl.dat',
                            world_size=world_size, rank=rank)
    train_run(config, split, job_dir, device)
    dist.destroy_process_group()


def dist_training(distributed, config: Config, split: Split, job_dir: Path):
    mp.spawn(process_run, args=(
        distributed, config, split, job_dir
    ), nprocs=len(distributed))


def single_training(device, config: Config, split: Split,
                    job_dir: Path):
    return train_run(config, split, job_dir, device)
