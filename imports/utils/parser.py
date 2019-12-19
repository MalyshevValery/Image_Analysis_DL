"""Settings parser"""
import copy
import datetime
import json
import os

from albumentations import Compose
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from imports.data import AlbumentationsWrapper, Loader
from imports.models.unet import UNet
from .maps import *


class SettingsParser:
    """This class parses settings.json"""

    def __init__(self, json_filename, predict_mode=False):
        with open(json_filename, 'r') as file:
            settings = json.load(file)
            self.settings = copy.deepcopy(settings)
        self.predict_mode = predict_mode

        # Augmentations
        self.augmentation_all = AlbumentationsWrapper.from_json(settings['augmentation_all'])
        self.augmentation_train = AlbumentationsWrapper.from_json(settings['augmentation_train'])
        self.augmentation_train_merged = Compose([self.augmentation_train, self.augmentation_all])

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
            if 'loss' in self.model_compile['metrics']:
                self.model_compile['metrics'].remove('loss')
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

        # Other params
        self.predict = settings['predict'] if 'predict' in settings else False
        self.show_sample = settings['show_sample'] if 'show_sample' in settings else False

        # Utility data
        general_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        settings_name = os.path.basename(json_filename)
        if settings_name.startswith('settings_'):
            settings_name = settings_name[len('settings_'):]
        if settings_name.endswith('.json'):
            settings_name = settings_name[:-len('.json')]

        self.results_dir = os.path.join("Jobs", general_name + '_' + self.model + '_' + settings_name)
        if not os.path.exists(self.results_dir) and not self.predict_mode:
            os.makedirs(self.results_dir)

    def get_loader(self):
        return Loader.from_json(self.settings['loader'])

    def get_model_method(self):
        """Returns method for model creation according to model.name setting"""
        if self.model == 'unet':
            return UNet
        else:
            raise Exception("Unknown model name")

    def get_callbacks(self):
        """Get callbacks for training"""
        callbacks = []
        metric_name = self.metrics_names[0]
        mode = 'max' if not metric_name == 'loss' else 'min'
        for s in self.callbacks_names:
            if s == "early_stop":
                callbacks.append(
                    EarlyStopping(monitor='val_' + metric_name, verbose=1, min_delta=0.01, patience=3, mode=mode,
                                  restore_best_weights=True))
            elif s == "tensorboard":
                callbacks.append(TensorBoard(log_dir=self.results_dir, profile_batch=0))
            elif s == "checkpoint":
                callbacks.append(
                    ModelCheckpoint(os.path.join(self.results_dir, 'weights.h5'), monitor='val_' + metric_name,
                                    verbose=1, save_best_only=True, mode=mode))
            elif s == 'keep_settings':
                self.keep_settings()
        return callbacks

    def keep_settings(self):
        """Dumps settings to folder with models to be able to reproduce results later"""
        if self.predict_mode:
            return
        with open(os.path.join(self.results_dir, 'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=2)
