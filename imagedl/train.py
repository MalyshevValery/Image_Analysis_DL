"""Run test features net"""

import logging
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from imagedl.config import Config
from imagedl.data import Splitter, Split
from .train_utils import dist_training, single_training
from .utility_config import DEVICE, DISTRIBUTED


def train(config: Config, split: Split,
          job_dir: Path, own_split: bool = False) -> Optional[pd.DataFrame]:
    """Function for one train run (use main_train to prepare logging and
    prepare arguments

    :param config: Main configuration object
    :param split: Data split
    :param job_dir: Job Dir (Path)
    :param own_split: True if you're loading from checkpoint and want to
        change data split
    :return: DataFrame with test metrics result
    """
    try:
        if DISTRIBUTED is None:
            return single_training(DEVICE, config, split, job_dir, own_split)
        else:
            return dist_training(DISTRIBUTED, config, split, job_dir)
    except Exception as e:
        traceback.print_exc()
        logging.error(e)
    return None


def main_train(config: Config, train_: float, val: float, test: float,
               kfold: int = None) -> None:
    """Main training procedure

    :param config: Configuration object
    :param train_: Fracture of train data
    :param val: Fracture of validation data
    :param test: Fracture of test data
    :param kfold: NUmber of folds, or None if not used
    """
    dataset, groups, *_, split = config.data
    job_dir = config.job_dir
    fmt = '[%(levelname)s] %(asctime)s %(message)s'
    logging.basicConfig(filename=str(job_dir / 'train.log'),
                        level=logging.INFO, format=fmt)

    print(f'Total number of samples: {len(dataset)}')
    logging.info(f'Total number of samples: {len(dataset)}')
    splitter = Splitter(total=len(dataset), group_labels=groups)

    if kfold is None:
        own_split = True
        if split is None:
            own_split = False
            split = splitter.random_split((train_, val, test))
        train(config, split, job_dir, own_split)
    else:
        frames = []
        for i, split in enumerate(
                splitter.k_fold(val, kfold)):
            print(f'Fold {i + 1}')
            logging.info(f'Fold {i + 1}')
            fold_job_dir = job_dir / f'Fold_{i + 1}'
            config.job_dir = fold_job_dir
            frames.append(train(config, split, fold_job_dir))
        overall = pd.concat(frames, ignore_index=True)
        print('All folds')
        print(overall)
        overall.to_csv(f'{job_dir}/metrics.csv')
        print('MEAN')
        print(overall.mean())
        logging.info(overall.mean())
