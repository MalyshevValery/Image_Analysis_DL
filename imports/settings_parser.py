import datetime
import json
import os
import copy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from imports.cnn.architectures.unet import UNet
from imports.cnn.metrics import iou
from imports.data_generators.image_mask_generator import ImageMaskGenerator

# Map for metrics labels
from imports.data_generators.image_reg_mask_generator import ImageRegMaskGenerator

metrics_map = {
    'acc': 'acc',
    'iou': iou
}

mode_map = {
    'loss': 'min',
    'acc': 'max',
    'iou': 'max'
}


class SettingsParser:
    """This class parses settings.json"""

    def __init__(self, json_filename):
        with open(json_filename, 'r') as file:
            settings = json.load(file)
            self.settings = copy.deepcopy(settings)

        self.images_path = settings['data']['images']
        self.masks_path = settings['data']['masks']
        self.descriptor_path = settings['data']['descriptor']
        self.reg_path = settings['data']['reg']

        self.gen_type = settings['generator_type']
        if 'generator' in settings:
            self.generator_args = settings['generator']
        else:
            self.generator_args = {}

        if 'registration' in settings:
            self.registration_args = settings['registration']
        else:
            self.registration_args = {}

        self.model = settings['model']['name']
        self.model_params = settings['model']
        del self.model_params['name']

        self.model_compile = settings['model_compile']
        if 'metrics' in self.model_compile:
            self.metrics_names = self.model_compile['metrics'].copy()
            self.model_compile['metrics'] = list(map(lambda s: metrics_map[s], self.model_compile['metrics']))
        else:
            self.metrics_names = []

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

        if 'predict' in settings:
            self.predict = settings['predict']
        else:
            self.predict = False

        self.general_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-" + self.model)
        self.results_dir = os.path.join("Jobs", self.general_name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def get_data_generator(self):
        """Returns generator object created according to settings.json and input shape for it"""
        if self.gen_type == 'norm':
            return ImageMaskGenerator(self.images_path, self.masks_path, **self.generator_args), (256, 256, 3)
        elif self.gen_type == 'reg':
            return ImageRegMaskGenerator(self.images_path, self.masks_path, self.reg_path, self.descriptor_path,
                                         **self.generator_args,
                                         **self.registration_args), (256, 256, 2)
        else:
            raise Exception('Unknown generator type')

    def get_model_method(self):
        """Returns method for model creation according to model.name setting"""
        if self.model == 'unet':
            return UNet
        else:
            raise Exception("Unknown model name")

    def get_callbacks(self):
        """Makes callbacks from labels in settings.json

        Possible values:
            - early_stop
            - tensorboard
            - checkpoint
        """
        callbacks = []
        for s in self.callbacks_names:
            if s == "early_stop":
                callbacks.append(
                    EarlyStopping(monitor='val_' + self.metrics_names[0], verbose=1, min_delta=0.01,
                                  patience=3, mode=mode_map[self.metrics_names[0]], restore_best_weights=True))
            elif s == "tensorboard":
                callbacks.append(TensorBoard(log_dir=self.results_dir, profile_batch=0))
            elif s == "checkpoint":
                callbacks.append(
                    ModelCheckpoint(os.path.join(self.results_dir, 'weights.h5'), monitor='val_' + self.metrics_names[0],
                                    verbose=1, save_best_only=True, mode=mode_map[self.metrics_names[0]]))
            elif s == 'keep_settings':
                self.keep_settings()
        return callbacks

    def keep_settings(self):
        """Dumps settings to folder with models to be able to reproduce results later"""
        with open(os.path.join(self.results_dir, 'settings.json'), 'w') as file:
            json.dump(self.settings, file, indent=2)
