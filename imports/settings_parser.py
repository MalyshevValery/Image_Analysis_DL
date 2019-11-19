import datetime
import json
import os
import copy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from imports.cnn.architectures.unet import UNet
from imports.cnn.metrics import iou
from imports.data_generators.image_mask_generator import ImageMaskGenerator

# Map for metrics labels
metrics_map = {
    'acc': 'acc',
    'iou': iou
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

        self.generator_args = settings['generator']
        self.registration_args = settings['registration']

        self.model = settings['model']['name']
        self.model_params = settings['model']
        del self.model_params['name']

        self.model_compile = settings['model_compile']
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
            - data_args
                - images_path
                - masks_path
        Optional fields:
            according to DataGenerator class constructor
        """
        return ImageMaskGenerator(self.images_path, self.masks_path, **self.generator_args)

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
                    EarlyStopping(monitor=self.model_compile['metrics'][0], verbose=1, min_delta=0.01, patience=3,
                                  mode='max', restore_best_weights=True))
            elif s == "tensorboard":
                log_dir = "Logs/" + self.general_name
                callbacks.append(TensorBoard(log_dir=log_dir, profile_batch=0))
            elif s == "checkpoint":
                if not os.path.exists('Models'):
                    os.makedirs('Models')
                callbacks.append(
                    ModelCheckpoint('Models/' + self.general_name + '.h5', monitor=self.model_compile['metrics'][0],
                                    verbose=1, save_best_only=True, mode='max'))
                self.keep_settings()
        return callbacks

    def keep_settings(self):
        """Dumps settings to folder with models to be able to reproduce results later"""
        with open('Models/' + self.general_name + '-settings.json', 'w') as file:
            json.dump(self.settings, file)
