import logging
from typing import Iterable

import torch
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from imagedl.data.datasets import SubDataset
from imagedl.utility_config import WORKERS


def prepare_batch(batch, device=None, non_blocking=False):
    new_batch = []
    for b in batch:
        if isinstance(b, torch.Tensor):
            new_batch.append(convert_tensor(b, device, non_blocking))
        elif isinstance(b, dict):
            new_b = {}
            for k in b.keys():
                new_b[k] = convert_tensor(b[k], device, non_blocking)
            new_batch.append(new_b)
        elif isinstance(b, Iterable):
            new_batch.append(
                [convert_tensor(b_i, device, non_blocking) for b_i in b])
        else:
            raise ValueError('Wrong data format')

    return tuple(new_batch)


def get_data_loaders(config, split):
    epochs, batch_size, test_batch_size, patience = config.train
    dataset, _, train_transform, test_transform, train_sampler, _ = config.data
    train_ds = SubDataset(dataset, split.train, transform=train_transform)
    val_ds = SubDataset(dataset, split.val, transform=test_transform)
    test_ds = SubDataset(dataset, split.test, transform=test_transform)

    info_str = f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}'
    print(info_str)
    logging.info(info_str)

    # if DISTRIBUTED is not None:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_ds)
    # else:
    #     train_sampler = None
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
