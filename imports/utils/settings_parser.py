import datetime
import json
import os
import copy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from imports.cnn.models.unet import UNet
from imports.utils.settings_maps import *


class SettingsParser:
    """This class parses settings.json"""

    def __init__(self, json_filename, predict_mode=False):
        with open(json_filename, 'r') as file:
            settings = json.load(file)
            self.settings = copy.deepcopy(settings)
        self.predict_mode = predict_mode

        # Data
        self.images_path = settings['data']['images']
        self.masks_path = settings['data']['masks']
        self.descriptor_path = settings['data']['descriptor']
        self.reg_path = settings['data']['reg']
        self.input_shape = settings['data']['input_shape']

        # Loader
        self.loader_type = settings['loader_type']
        self.loader_decorators = settings['loader_decorators'] if 'loader_decorators' in settings else []
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

        # Other params
        self.predict = settings['predict'] if 'predict' in settings else False
        self.show_sample = settings['show_sample'] if 'show_sample' in settings else False

        # Utility data
        general_name = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        settings_name = os.path.basename(json_filename)
        if settings_name.startswith('settings_'):
            settings_name = settings_name[len('settings_'):]
        if settings_name.endswith('.json'):
            settings_name = settings_name[:-len('.json')]

        self.results_dir = os.path.join("Jobs", general_name + '_' + self.model + '_' + settings_name)
        if not os.path.exists(self.results_dir) and not self.predict_mode:
            os.makedirs(self.results_dir)

    def get_loader(self):
        """Returns decorated loader object created according to settings.json and input shape for it"""
        loader_cls = loader_class[self.loader_type]
        for d in self.loader_decorators:
            t_dec = d.copy()
            dec = decorators[t_dec['name']]
            del t_dec['name']
            loader_cls = dec(loader_cls, **t_dec)

        if self.loader_type == 'norm':
            return loader_cls(self.images_path, self.masks_path, **self.loader_args)
        elif self.loader_type == 'reg':
            return loader_cls(self.images_path, self.masks_path, self.reg_path, self.descriptor_path,
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
        if self.predict_mode:
            return
        with open(os.path.join(self.results_dir, 'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=2)
