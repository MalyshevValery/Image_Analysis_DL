"""Class that encapsulates training process"""
import copy
import functools
import os

import numpy as np
import tensorflow as tf
from albumentations import BasicTransform, Compose
from tensorflow.keras.models import Model

from .data import Loader, image_mask, MaskGenerator, AbstractStorage


class TrainWrapper:
    """This class encapsulate model training process.
    It allows training, evaluation as well as checking is the configuration valid."""

    def __init__(self, loader: Loader, model: Model, job_dir: str,
                 train_val_test=(0.8, 0.1, 0.1), batch_size=1,
                 restore_weights=True,
                 augmentation_train: BasicTransform = None,
                 augmentation_all: BasicTransform = None,
                 generator_params: dict = None, eval_metric='loss'):
        """Constructor

        :param loader: Loader for data
        :param model: Model
        :param job_dir: directory to save model, events, predictions
        :param train_val_test: split on training, validation and test sets
        :param batch_size: batch size
        :param restore_weights: restores weights from "weights.h5" after training if True
        :param augmentation_train: augmentations for train data (will be merged with augmentations_all)
        :param augmentation_all: augmentations applied to data from all sets
        :param generator_params: parameters for training
        :param eval_metric: metric for hyper parameter evaluation
        """
        self._loader = loader
        self._train_val_test = train_val_test
        self._batch_size = batch_size
        self._augmentation_train = augmentation_train
        self._augmentation_all = augmentation_all
        self._restore_weights = restore_weights
        self._eval_metric = eval_metric
        self._composed_augmentation = self._augmentation_train
        if self._augmentation_all and self._augmentation_train:
            self._composed_augmentation = Compose(
                [self._augmentation_train, self._augmentation_all])
        elif self._augmentation_all:
            self._composed_augmentation = self._augmentation_all

        train_keys, val_keys, test_keys = loader.split(train_val_test)
        self._train_gen = MaskGenerator(train_keys, loader, batch_size,
                                        self._composed_augmentation)
        self._val_gen = MaskGenerator(val_keys, loader, batch_size,
                                      shuffle=False,
                                      augmentations=self._augmentation_all)
        self._test_gen = MaskGenerator(test_keys, loader, batch_size,
                                       shuffle=False,
                                       augmentations=self._augmentation_all)

        # Training params and callbacks
        self._generator_params = generator_params or {}

        # # Create model
        # if not isinstance(model, Model):
        #     raise TypeError('Provided model is not a keras Model')

        self._input_shape = loader.get_input_shape()
        # if model.input_shape != (None, *self._input_shape):
        #     raise ValueError('Shape of model ' + str(model.input_shape) +
        #                      ' is different from shape of data ' + str(
        #         self._input_shape))
        self._model = model

        self._job_dir = job_dir
        if not os.path.exists(job_dir):
            os.makedirs(self._job_dir)
        elif not os.path.isdir(job_dir):
            raise FileExistsError(
                job_dir + ' file exists, directory cannot be created')
        self._metrics = []

    @functools.wraps(tf.keras.Model.compile)
    def compile(self, **kwargs):
        if 'metrics' in kwargs:
            self._metrics = kwargs['metrics']
        self._model.compile(**kwargs)
        self._model.summary()

    def train(self, callbacks, restore_weights=True):
        """Train process. After training best weights are restored if checkpoint
        callback with save_best was used"""
        self._model.fit(self._train_gen,
                        validation_data=self._val_gen,
                        callbacks=callbacks,
                        **self._generator_params)
        self._model.save(os.path.join(self._job_dir, 'model.h5'))
        if restore_weights:
            self._model.load_weights(
                os.path.join(self._job_dir, "best_model.h5"))

    def evaluate(self):
        """Evaluate test set and returns value of selected metric"""
        ret = self._model.evaluate(self._test_gen,
                                   **self.__get_eval_params())
        metrics = ['loss'] + [m if isinstance(m, str) else m.name for m in
                              self._metrics]
        ret_val = zip(metrics, ret)
        # for entry in ret_val:
        #     open(os.path.join(self._job_dir, '%.3f_' % entry[1] + entry[0]),
        #          'w')
        k = 1
        eval_metric = self._eval_metric
        if self._eval_metric[0] == '-':
            k = -1
            eval_metric = self._eval_metric[1:]
        return ret[metrics.index(eval_metric)] * k

    def check(self):
        """Checks if everything in configuration is valid by running one batch through model"""
        batches = [self._train_gen[0], self._val_gen[0], self._test_gen[0]]
        x = [b[0] for b in batches]
        y = [b[1] for b in batches]
        self._model.train_on_batch(x[0], y[0])
        self._model.test_on_batch(x[1], y[1])
        self._model.predict_on_batch(x[2])

    def predict_save_test(self, storage):
        """Predicts test data and saves it to given storage"""
        if not isinstance(storage, AbstractStorage):
            raise TypeError('storage is not an instance of', AbstractStorage)
        predicted = self._model.predict(self._test_gen,
                                        **self.__get_eval_params())
        self._loader.save_predicted(self._test_gen.keys, predicted, storage)

    def get_train_sample(self):
        """Returns random image with ground truth overlay from train generator"""
        to_show = TrainWrapper.__get_sample(self._train_gen)
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
        """Transforms model training fit_generator parameters to the ones used by valuate_generator or
        predict_generator. This is done by removing some key-value pairs from parameter dictionary
        """
        evaluate_params = copy.deepcopy(self._generator_params)
        evaluate_params.pop('epochs', None)
        return evaluate_params
