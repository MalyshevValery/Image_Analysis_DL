"""Data related training stuff"""
from collections import Sequence

import torch
from ignite.utils import convert_tensor
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

from imagedl import Config
from imagedl.data import Split
from imagedl.utility_config import WORKERS
from imagedl.utils.types import DataType, T_co
from .logger import info
from ..data.datasets import MapDataset


def prepare_batch(batch: DataType, device: torch.device = None,
                  non_blocking: bool = False) -> DataType:
    """Prepares batch (puts on right device for now)

    :param batch: Can be any combination of tuples, dicts and tensors
    :param device: device to put data in
    :param non_blocking: Wait till copy finished or no
    :return:
    """
    if isinstance(batch, torch.Tensor):
        ret: Tensor = convert_tensor(batch, device, non_blocking)
        return ret
    elif isinstance(batch, dict):
        return {k: prepare_batch(v, device, non_blocking) for k, v in
                batch.items()}
    elif isinstance(batch, Sequence):
        return tuple(
            [prepare_batch(b, device, non_blocking) for b in batch])
    else:
        raise ValueError('Wrong data format')


def get_data_loaders(config: Config,
                     split: Split) -> Sequence[DataLoader[T_co]]:
    """Create data loaders and visualize training samples"""
    epochs, batch_size, test_batch_size, patience = config.train
    dataset, _, train_transform, test_transform, train_sampler, _ = config.data
    train_ds = MapDataset(Subset(dataset, list(split.train)), train_transform)
    val_ds = MapDataset(Subset(dataset, list(split.val)), test_transform)
    test_ds = MapDataset(Subset(dataset, list(split.test)), test_transform)

    info_str = f'Train: {len(train_ds)}, Val: {len(val_ds)}'
    info_str += f', Test: {len(test_ds)}'
    info(info_str)

    train_dl = DataLoader(train_ds, batch_size, num_workers=WORKERS,
                          sampler=train_sampler(
                              train_ds) if train_sampler is not None else None,
                          shuffle=train_sampler is None)
    val_dl = DataLoader(val_ds, test_batch_size, num_workers=WORKERS,
                        shuffle=True)
    test_dl = DataLoader(test_ds, test_batch_size, num_workers=WORKERS)

    img = config.visualize(*next(iter(train_dl)))
    config.save_sample(img, config.job_dir / 'train')
    return train_dl, val_dl, test_dl
