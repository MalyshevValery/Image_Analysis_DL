"""Class that encapsulates training process"""
import copy
import datetime
import os

import numpy as np
from albumentations import BasicTransform, Compose
from tensorflow.keras.models import Model, save_model

from .data import Loader, image_mask, MaskGenerator, AbstractStorage
from .jsonserializable import JSONSerializable
from .models import ModelsFactory
from .wrappers import CallbacksWrapper, CompileParams, AlbumentationsWrapper


class TrainingWrapper(JSONSerializable):
    """Training process encapsulation"""

    def __init__(self, loader: Loader, model: Model, job_dir: str, model_compile: CompileParams = None,
                 train_val_test=(0.8, 0.1, 0.1), batch_size=1, restore_weights=True,
                 augmentation_train: BasicTransform = None, augmentation_all: BasicTransform = None,
                 callbacks: CallbacksWrapper = None, training_params: dict = None):
        """Constructor

        :param loader: Loader for data
        :param model: Model
        :param job_dir: directory to save model, events, predictions
        :param model_compile: model compilation parameters
        :param train_val_test: split on training, validation and test sets
        :param batch_size: batch size
        :param restore_weights: restores weights from "weights.h5" after training if True
        :param augmentation_train: augmentations for train data (will be merged with augmentations_all)
        :param augmentation_all: augmentations applied to data from all sets
        :param callbacks: callbacks for training process
        :param training_params: parameters for training
        """
        self._loader = loader
        self._train_val_test = train_val_test
        self._batch_size = batch_size
        self._augmentation_train = augmentation_train
        self._augmentation_all = augmentation_all
        self._restore_weights = restore_weights
        self._composed_augmentation = self._augmentation_train
        if self._augmentation_all and self._augmentation_train:
            self._composed_augmentation = Compose([self._augmentation_train, self._augmentation_all])
        elif self._augmentation_all:
            self._composed_augmentation = self._augmentation_all

        train_keys, val_keys, test_keys = loader.split(train_val_test)
        self._train_gen = MaskGenerator(train_keys, loader, batch_size, self._composed_augmentation)
        self._val_gen = MaskGenerator(val_keys, loader, batch_size, shuffle=False, augmentations=self._augmentation_all)
        self._test_gen = MaskGenerator(test_keys, loader, batch_size, shuffle=False,
                                       augmentations=self._augmentation_all)

        # Training params and callbacks
        self._callbacks = callbacks or None
        self._training_params = training_params or {}

        # Create model
        if not isinstance(model, Model):
            raise TypeError('Provided model is not a keras Model')

        self._input_shape = loader.get_input_shape()
        if model.input_shape != (None, *self._input_shape):
            raise ValueError('Shape of model ' + str(model.input_shape) +
                             ' is different from shape of data ' + str(self._input_shape))
        self._model = model
        self._model_compile = model_compile or CompileParams({})
        model.compile(**self._model_compile.get_params())
        model.summary()

        # Check job_dir
        self._job_dir = job_dir
        if not os.path.exists(job_dir):
            os.makedirs(self._job_dir)
        elif not os.path.isdir(job_dir):
            raise FileExistsError(job_dir + ' file exists, directory cannot be created')

    def train(self, save_whole_model=False):
        """Train process. After training best weights are restored if checkpoint callback with save_best was used"""
        self._model.fit_generator(self._train_gen, validation_data=self._val_gen,
                                  callbacks=self._callbacks.get_callbacks() if self._callbacks else None,
                                  **self._training_params)
        self._model.save_weights(os.path.join(self._job_dir, 'weights_last.h5'))
        if self._restore_weights:
            self._model.load_weights(os.path.join(self._job_dir, "weights.h5"))

        if save_whole_model:
            save_model(self._model, os.path.join(self._job_dir, 'model.h5'))

    def evaluate(self):
        """Evaluate test set"""
        ret = self._model.evaluate_generator(self._test_gen, **self.__get_eval_params())
        ret_val = {'loss': ret[0]}
        if len(ret) > 1 and 'metrics' in self._training_params:
            for i, metric in enumerate([metric if isinstance(metric, str) else metric.name for metric in
                                        self._training_params['metrics']]):
                if metric == 'loss':
                    continue
                ret_val[metric] = ret[i + 1]
        print('Test results: ', ret_val)
        for key in ret_val:
            open(os.path.join(self._job_dir, '%.3f_' % ret_val[key] + key), 'w')

    def check(self):
        """Checks if anything is valid by running one batch through model"""
        batches = [self._train_gen[0], self._val_gen[0], self._test_gen[0]]
        x = [b[0] for b in batches]
        y = [b[1] for b in batches]
        self._model.train_on_batch(x[0], y[0])
        self._model.test_on_batch(x[1], y[1])
        self._model.predict_on_batch(x[2])

    def predict_save_test(self, storage):
        """Predicts test data"""
        if not isinstance(storage, AbstractStorage):
            raise TypeError('storage is not an instance of', AbstractStorage)
        predicted = self._model.predict_generator(self._test_gen, **self.__get_eval_params())
        self._loader.save_predicted(self._test_gen.keys, predicted, storage)

    def to_json(self):
        """Saves training process config to dict"""
        config = {
            'loader': self._loader.to_json(),
            'train_val_test': self._train_val_test,
            'batch_size': self._batch_size,
            'model': ModelsFactory.to_json(self._model),
            'job_dir': os.path.abspath(self._job_dir),
            'model_compile': self._model_compile.to_json(),
            'training_params': self._training_params,
            'callbacks': self._callbacks.to_json()
        }
        if self._augmentation_all:
            config['augmentation_all'] = AlbumentationsWrapper.to_json(self._augmentation_all),
        if self._augmentation_all:
            config['augmentation_train'] = AlbumentationsWrapper.to_json(self._augmentation_all),
        if self._augmentation_all:
            config['callbacks'] = self._callbacks.to_json()
        return config

    @staticmethod
    def from_json(json, job_prefix=None, settings_filename='settings.json'):
        """Restores training configuration to dict"""
        config = copy.deepcopy(json)
        loader = Loader.from_json(config['loader'])
        del config['loader']
        model = ModelsFactory.from_json(config['model'],
                                        None if 'input_shape' in config['model'] else loader.get_input_shape())
        del config['model']
        if 'job_dir' not in config:
            general_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            settings_name = os.path.basename(settings_filename)
            if settings_name.startswith('settings'):
                settings_name = settings_name[len('settings'):]
            if settings_name.endswith('.json'):
                settings_name = settings_name[:-len('.json')]
            job_dir = general_name + '_' + model.name
            if settings_name != '':
                job_dir += '_' + settings_name
            if job_prefix is not None:
                job_dir = job_prefix + job_dir

        else:
            job_dir = config['job_dir']
            del config['job_dir']

        if 'augmentation_train' in config:
            config['augmentation_train'] = AlbumentationsWrapper.from_json(config['augmentation_train'])
        if 'augmentation_all' in config:
            config['augmentation_all'] = AlbumentationsWrapper.from_json(config['augmentation_all'])
        if 'callbacks' in config:
            config['callbacks'] = CallbacksWrapper.from_json(config['callbacks'], job_dir)
        if 'model_compile' in config:
            config['model_compile'] = CompileParams.from_json(config['model_compile'])

        return TrainingWrapper(loader, model, job_dir, **config)

    def get_train_sample(self):
        """Returns random image with ground truth overlay from train generator"""
        to_show = TrainingWrapper.__get_sample(self._train_gen)
        image = to_show[0]
        mask = to_show[1]
        return image_mask(image, mask)

    def get_job_dir(self):
        """Returns path to job dir"""
        return os.path.abspath(self._job_dir)

    @staticmethod
    def __get_sample(gen):
        batch = gen[np.random.randint(0, len(gen))]
        k = np.random.randint(len(batch[0]))
        sample = [b[k] for b in batch]
        return sample

    def __get_eval_params(self):
        evaluate_params = copy.deepcopy(self._training_params)
        evaluate_params.pop('epochs', None)
        return evaluate_params
