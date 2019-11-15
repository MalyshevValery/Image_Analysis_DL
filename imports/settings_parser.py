import datetime
import json
import os
import copy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from imports.cnn.architectures.unet import UNet
from imports.cnn.metrics import iou
from imports.data_generator import DataGenerator

# Map for metrics labels
metrics_map = {
    'acc': 'acc',
    'iou': iou
}

max_min_map = {
    'acc': 'max',
    'iou': 'max',
    'loss': 'min'
}


class SettingsParser:
    """This class parses settings.json"""
    def __init__(self, json_filename):
        with open(json_filename, 'r') as file:
            settings = json.load(file)
            self.settings = copy.deepcopy(settings)

        self.generator_args = settings['generator_args']
        self.model = settings['model']['name']
        self.model_params = settings['model']
        del self.model_params['name']

        self.model_compile = settings['model_compile']
        self.metrics_names = self.model_compile['metrics'].copy()
        self.model_compile['metrics'] = list(map(lambda s: metrics_map[s], self.model_compile['metrics']))

        training = settings['training']
        try:
            self.callbacks_names = training['callbacks']
        except Exception:
            self.callbacks_names = []

        try:
            self.batch_size = training['batch_size']
        except Exception:
            self.batch_size = 1

        self.epochs = training['epochs']

        self.general_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-" + self.model)

    def get_data_generator(self):
        """Returns generator object created according to settings.json

        Required fields:
            - images_path
            - masks_path
        Optional fields:
            according to DataGenerator class constructor
        """
        copy = self.generator_args.copy()
        images_path = self.generator_args['images_path']
        masks_path = self.generator_args['masks_path']
        del copy['images_path']
        del copy['masks_path']
        return DataGenerator(images_path, masks_path, **copy)

    def get_model_method(self):
        """Returns method for model creation according to model.name setting"""
        if self.model == 'unet':
            return UNet
        else:
            print("Unknown model name")
            print("Possible model names: unet")
            return None

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
                    EarlyStopping(monitor='val_' + self.metrics_names[0],
                        verbose=1, min_delta=0.01,
                        patience=3,mode=max_min_map[self.metrics_names[0]],
                        restore_best_weights=True))
            elif s == "tensorboard":
                log_dir = "Logs/" + self.general_name
                callbacks.append(TensorBoard(log_dir=log_dir, profile_batch=0))
            elif s == "checkpoint":
                if not os.path.exists('Models'):
                    os.makedirs('Models')
                callbacks.append(
                    ModelCheckpoint('Models/' + self.general_name + '.h5',
                        monitor='val_'+self.metrics_names[0],
                        verbose=1, save_best_only=True,
                        mode=max_min_map[self.metrics_names[0]]))
                self.keep_settings()
        return callbacks

    def keep_settings(self):
        """Dumps settings to folder with models to be able to reproduce results later"""
        with open('Models/' + self.general_name + '-settings.json', 'w') as file:
            json.dump(self.settings, file)