import os
import sys

import numpy as np
from albumentations import ToFloat
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, \
    EarlyStopping

from imports import TrainWrapper
from imports.data import Loader
from imports.data.extensions import TypeScaleExtension, SplitMaskExtension, \
    IgnoreRegionExtension
from imports.models import UNet

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from imports.data.storages import DirectoryStorage, Mode
from segmentation_models.metrics import IOUScore
from segmentation_models.losses import DiceLoss

if __name__ == '__main__':
    images = DirectoryStorage('Data/Test_Images', gray_transform=True)
    masks = DirectoryStorage('Data/Masks', gray_transform=True)

    extensions = {
        'mask': [TypeScaleExtension(src_max=255, target_type=np.float32,
                                    target_max=1.0),
                 SplitMaskExtension(),
                 IgnoreRegionExtension()
                 ],
        'save_image': TypeScaleExtension(255, np.float32, 1),
        'save': TypeScaleExtension(1, np.uint8, 255)
    }
    loader = Loader(images, masks, extensions)
    model = UNet(input_shape=(256, 256, 1))
    trainer = TrainWrapper(loader, model, 'Jobs',
                           train_val_test=(0.8, 0.1, 0.1),
                           batch_size=4, eval_metric=IOUScore().name,
                           augmentation_all=ToFloat(),
                           generator_params={
                               'epochs': 5,
                               'workers': 1
                           })
    trainer.compile(optimizer='adam', loss=DiceLoss(),
                    metrics=['acc', IOUScore()])

    callbacks = [
        ModelCheckpoint('Jobs/best_model.h5'),
        TensorBoard(log_dir='Jobs', profile_batch=0),
        EarlyStopping(patience=2, restore_best_weights=True)
    ]
    trainer.train(callbacks=callbacks, restore_weights=True)

    ret_val = trainer.evaluate()

    pred_dir = os.path.join(trainer.get_job_dir(), 'predicted')
    trainer.predict_save_test(DirectoryStorage(pred_dir, mode=Mode.WRITE))
    print('TEST EVALUATION METRIC: ', ret_val)
