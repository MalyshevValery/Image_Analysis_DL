"""Run test features net"""
import json
import pandas as pd
import click
from typing import NamedTuple, Dict

import skimage.io as io
import numpy as np
from datetime import datetime

from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping
)

import tensorflow as tf
from segmentation_models.metrics import IOUScore
from segmentation_models.losses import DiceLoss

import sys
import os

from tensorflow_addons.optimizers import RectifiedAdam

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from imports import TrainWrapper
from imports.data import Loader, AugmentationMap, IO, Splitter, Split
from imports.data.extensions import TypeScaleExtension, ChannelExtension
from imports.data.storages import HDF5Storage, ComposeStorage, MockStorage, \
    DirectoryStorage
from imports.models import UNet
from imports.utils import RunParams
import albumentations as alb

ALPHA = 0.4
AUG_P = 0.3
MAIN_DIR = '/home/malyshev/Projects/Data_CAM17'


class TrainingObjects(NamedTuple):
    """Objects required for training"""
    loader: Loader
    params: RunParams
    model: tf.keras.Model


def train(objects: TrainingObjects,
          job_dir: str, split: Split) -> pd.DataFrame:
    """Training procedure depending on split"""

    trainer = TrainWrapper(objects.loader, objects.model,
                           job_dir, split,
                           generator_params=objects.params)

    big_image = np.zeros((2560, 2560, 3))
    itr = iter(split.train)
    try:
        for i in range(10):
            for j in range(10):
                batch = next(itr)
                image = batch[0][0][0]
                mask = batch[1][0][0][..., 0]
                image[..., 2] *= 1 - ALPHA * mask
                big_image[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = image
    except Exception:
        pass
    io.imsave(f'{job_dir}/aug.png', big_image)

    callbacks = [
        EarlyStopping(patience=10, mode='max', monitor='val_iou_score'),
        TensorBoard(profile_batch=0, log_dir=job_dir),
        ModelCheckpoint(f'{job_dir}/best-model.h5', save_best_only=True,
                        verbose=1, mode='max', monitor='val_iou_score')
    ]

    with open(f'{job_dir}/config.json', 'w') as file:
        json.dump(trainer.to_json(), file, indent=2)
    #trainer.train(callbacks, f'{job_dir}/best-model.h5')

    # Predict test set
    predicted = trainer.predict(split.test)
    assert predicted is not None
    predicted = predicted[..., 0]
    test_keys = split.test.keys
    test_images = objects.loader.get_input(test_keys)
    test_images = test_images / 255
    ground_truth = objects.loader.get_output(test_keys)[..., 0]

    test_images[..., 0] *= 1 - ALPHA * ground_truth
    test_images[..., 2] *= 1 - ALPHA * predicted

    storage = DirectoryStorage(f'{job_dir}/TestPred', writable=True)
    storage.save_array([k.replace('/', '_') for k in test_keys], test_images)

    df = pd.DataFrame(trainer.evaluate(), index=[0])
    df.to_csv(f'{job_dir}/metrics.csv')
    return df


@click.command()
@click.option('--train', '-t', 'train_', default=0.8, help='Training size',
              type=float)
@click.option('--val', '-v', default=0.1, help='Validation size', type=float)
@click.option('--test', '-s', default=0.1, help='Test size', type=float)
@click.option('--kfold', '-k', help='Number of folds', type=int)
def main(train_: float, val: float, test: float, kfold: int = None) -> None:
    """Main program"""
    prefixes = ['Tumour', 'NegHealth', 'PosHealth']
    filenames_l5 = [f'{MAIN_DIR}/{p}_l5.hdf5' for p in prefixes]

    image_storages = [HDF5Storage(f, 'images') for f in filenames_l5]

    train_aug = alb.Compose([
        alb.RGBShift(p=AUG_P, r_shift_limit=10, g_shift_limit=0,
                     b_shift_limit=10),
        alb.RandomBrightnessContrast(p=AUG_P),
        alb.Blur(p=AUG_P, blur_limit=3),
        alb.Compose([
            alb.VerticalFlip(),
            alb.HorizontalFlip(),
            alb.RandomRotate90(),
            alb.Transpose(),
        ], p=1),
        alb.ShiftScaleRotate(p=AUG_P),
        alb.GaussNoise(p=AUG_P),
    ], p=0)
    test_aug = alb.ToFloat(255)
    amap = AugmentationMap(image=(IO.INPUT, 0), mask=(IO.OUTPUT, 0))
    print(filenames_l5)

    mask_storage = HDF5Storage(filenames_l5[0], 'masks')
    dummy_storage1 = MockStorage(np.zeros((256, 256), np.uint8),
                                 image_storages[1].keys)
    dummy_storage2 = MockStorage(np.zeros((256, 256), np.uint8),
                                 image_storages[2].keys)

    input_storage = ComposeStorage(image_storages)
    output_storage = ComposeStorage(
        [mask_storage, dummy_storage1, dummy_storage2], extensions=(
            TypeScaleExtension(256, np.float32, 1.0),
            ChannelExtension(),
        ))

    loader = Loader(input_storage, output_storage)
    params = RunParams(epochs=1, shuffle=True)

    model = UNet((256, 256, 3))
    model.compile(RectifiedAdam(), DiceLoss(),
                          ['acc', IOUScore()])
    model.summary()
    config = TrainingObjects(
        loader=loader,
        params=params,
        model=model,
    )
    print(f'Data entries: {len(loader.keys)}')

    stamp = datetime.now()
    job_dir = f'Jobs/{stamp:%Y-%m-%d-%H-%M-%S}_UNet_CAM17'

    splitter = Splitter(loader, 4, train_aug, test_aug, amap)

    if kfold is None:
        split = splitter.random_split((train_, val, test),
                                      '\d*/(.*)_\d*_\d*\.png')
        train(config, job_dir, split)
    else:
        frames = []
        for i, split in enumerate(
                splitter.k_fold(val, kfold, '\d*/(.*)_\d*_\d*\.png')):
            tf.keras.backend.clear_session()
            print(f'Fold {i+1}')
            frames.append(train(config, f'{job_dir}_Fold_{i + 1}', split))
            print(frames[-1])
        overall = pd.concat(frames, ignore_index=True)
        print('All folds')
        print(overall)
        overall.to_csv(f'{job_dir}.csv')


if __name__ == '__main__':
    main()
