import datetime
import json
import os
import copy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from imports.cnn.models.unet import UNet

import segmentation_models.metrics as metrics
import segmentation_models.losses as losses

from imports.data.loaders import ImageRegMaskLoader, ImageMaskLoader
import albumentations as aug

metrics_map = {
    'iou': metrics.IOUScore(name='iou'),
    'f1': metrics.FScore(beta=1, name='f1'),
    'f2': metrics.FScore(beta=2, name='f2'),
    'precision': metrics.Precision(name='precision'),
    'recall': metrics.Recall(name='recall')
}

loss_map = {
    'jaccard': losses.JaccardLoss(),
    'dice': losses.DiceLoss(),
    'binary_focal': losses.BinaryFocalLoss()
}

augmentations = {
    'blur': aug.Blur,
    'bright_contrast': aug.RandomBrightnessContrast,
    'clahe': aug.CLAHE,
    'crop': aug.RandomCrop,
    'sized_crop': aug.RandomSizedCrop,
    'compression': aug.ImageCompression,
    'downscale': aug.Downscale,
    'equalize': aug.Equalize,
    'float': aug.ToFloat,
    'gaussian_noise': aug.GaussNoise,
    'gaussian_blur': aug.GaussianBlur,
    'iaa_sharpen': aug.IAASharpen,
    'iso_noise': aug.ISONoise,
    'median_blur': aug.MedianBlur,
    'resize': aug.Resize,
    'shift_scale_rotate': aug.ShiftScaleRotate
}


class SettingsParser:
    """This class parses settings.json"""

    def __init__(self, json_filename):
        with open(json_filename, 'r') as file:
            settings = json.load(file)
            self.settings = copy.deepcopy(settings)

        # Data
        self.images_path = settings['data']['images']
        self.masks_path = settings['data']['masks']
        self.descriptor_path = settings['data']['descriptor']
        self.reg_path = settings['data']['reg']
        self.input_shape = settings['data']['input_shape']

        # Loader
        self.loader_type = settings['loader_type']
        self.loader_args = settings['loader'] if 'loader' in settings else {}

        # Augmentations
        aug_names = [s['name'] for s in settings['aug_all']]
        aug_params = settings['aug_all']
        for a in aug_params:
            del a['name']
        self.aug_all = [augmentations[aug_names[i]](**aug_params[i]) for i in range(len(settings['aug_all']))]
        self.aug_all = aug.Compose(self.aug_all)
        print("Augmentations for all", self.aug_all)

        aug_names = [s['name'] for s in settings['aug_train']]
        aug_params = settings['aug_train']
        for a in aug_params:
            del a['name']
        self.aug_train = [augmentations[aug_names[i]](**aug_params[i]) for i in range(len(settings['aug_train']))]
        self.aug_train = aug.Compose(self.aug_train)
        self.aug_train = aug.Compose([self.aug_train, self.aug_all])
        print("Augmentations for train", self.aug_train)

        # Registration
        self.registration_args = settings['registration'] if 'registration' in settings else {}

        # Model
        self.model = settings['model']['name']
        self.model_params = settings['model']
        del self.model_params['name']

        # Model compile
        self.model_compile = settings['model_compile']
        loss = self.model_compile['loss']
        if loss in loss_map:
            self.model_compile['loss'] = loss_map[loss]

        if 'metrics' in self.model_compile:
            self.metrics_names = self.model_compile['metrics'].copy()
            self.model_compile['metrics'] = list(
                map(lambda s: metrics_map[s] if s in metrics_map else s, self.model_compile['metrics']))
        else:
            self.metrics_names = []

        # Training
        training = settings['training']
        if 'callbacks' in training:
            self.callbacks_names = training['callbacks']
            del training['callbacks']
        else:
            self.callbacks_names = []

        self.batch_size = training['batch_size']
        self.epochs = training['epochs']
        del training['batch_size']
        del training['epochs']

        self.training = training

        self.predict = settings['predict'] if 'predict' in settings else False
        self.show_sample = settings['show_sample'] if 'show_sample' in settings else False

        # Utility data
        self.general_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-" + self.model)
        self.results_dir = os.path.join("Jobs", self.general_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def get_loader(self):
        """Returns loader object created according to settings.json and input shape for it"""
        if self.loader_type == 'norm':
            return ImageMaskLoader(self.images_path, self.masks_path, **self.loader_args)
        elif self.loader_type == 'reg':
            return ImageRegMaskLoader(self.images_path, self.masks_path, self.reg_path, self.descriptor_path,
                                      **self.loader_args, **self.registration_args)
        else:
            raise Exception('Unknown loader type')

    def get_model_method(self):
        """Returns method for model creation according to model.name setting"""
        if self.model == 'unet':
            return UNet
        else:
            raise Exception("Unknown model name")

    def get_callbacks(self):
        callbacks = []
        for s in self.callbacks_names:
            if s == "early_stop":
                callbacks.append(
                    EarlyStopping(monitor='val_' + self.metrics_names[0], verbose=1, min_delta=0.01,
                                  patience=3, mode='max', restore_best_weights=True))
            elif s == "tensorboard":
                callbacks.append(TensorBoard(log_dir=self.results_dir, profile_batch=0))
            elif s == "checkpoint":
                callbacks.append(
                    ModelCheckpoint(os.path.join(self.results_dir, 'weights.h5'),
                                    monitor='val_' + self.metrics_names[0],
                                    verbose=1, save_best_only=True, mode='max'))
            elif s == 'keep_settings':
                self.keep_settings()
        return callbacks

    def keep_settings(self):
        """Dumps settings to folder with models to be able to reproduce results later"""
        with open(os.path.join(self.results_dir, 'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=2)
