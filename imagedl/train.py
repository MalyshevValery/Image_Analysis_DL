"""Run test features net"""
import pandas as pd

from imagedl.config import Config
from imagedl.data import Splitter
from .train_utils import dist_training, single_training
from .utility_config import DEVICE, DISTRIBUTED


def train(config: Config, split, job_dir):
    if DISTRIBUTED is None:
        single_training(DEVICE, config, split, job_dir)
    else:
        dist_training(DISTRIBUTED, config, split, job_dir)


def main_train(config: Config, train_: float, val: float, test: float,
               kfold: int = None) -> None:
    """Main train procedure"""
    dataset, groups, *_, split = config.data
    job_dir = config.job_dir

    print(f'Total number of samples: {len(dataset)}')
    splitter = Splitter(total=len(dataset), group_labels=groups)

    if kfold is None:
        if split is None:
            split = splitter.random_split((train_, val, test))
        train(config, split, job_dir)
    else:
        raise NotImplementedError()
        frames = []
        for i, split in enumerate(
                splitter.k_fold(val, kfold)):
            print(f'Fold {i + 1}')
            fold_job_dir = job_dir / f'Fold_{i + 1}'
            frames.append(train(config, split, fold_job_dir, DEVICE))
            print(frames[-1])
        print(frames)
        overall = pd.concat(frames, ignore_index=True)
        print('All folds')
        print(overall)
        overall.to_csv(f'{job_dir}/metrics.csv')
