"""Run test features net"""
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as distributed

from imagedl.config import Config
from imagedl.data import Splitter, Split
from .train_utils.data import get_data_loaders
from .train_utils.engines import evaluate, evaluator, create_trainer
from .train_utils.handlers import tensorboard_logger, train_handlers
from .utility_config import DEVICE, DISTRIBUTED


def train_run(config: Config, split: Split, job_dir: Path,
              device) -> pd.DataFrame:
    """Training procedure depending on split"""
    job_dir.mkdir(parents=True, exist_ok=True)
    model, optimizer, criterion, split, trainer = create_trainer(config, DEVICE,
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


def train(config: Config, train_: float, val: float, test: float,
          kfold: int = None) -> None:
    """Main train procedure"""
    if DISTRIBUTED:
        torch.cuda.set_device(DEVICE)
        torch.distributed.init_process_group("nccl",
                                             init_method="file:///tmp/nccl_dist.dat",
                                             world_size=len(DISTRIBUTED),
                                             rank=0)

    dataset, groups, _, _ = config.data
    job_dir = config.job_dir

    print(f'Total number of samples: {len(dataset)}')
    splitter = Splitter(total=len(dataset), group_labels=groups)

    if kfold is None:
        split = splitter.random_split((train_, val, test))
        train_run(config, split, job_dir, DEVICE)
    else:
        frames = []
        for i, split in enumerate(
                splitter.k_fold(val, kfold)):
            print(f'Fold {i + 1}')
            fold_job_dir = job_dir / f'Fold_{i + 1}'
            frames.append(train_run(config, split, fold_job_dir, DEVICE))
            print(frames[-1])
        print(frames)
        overall = pd.concat(frames, ignore_index=True)
        print('All folds')
        print(overall)
        overall.to_csv(f'{job_dir}/metrics.csv')
