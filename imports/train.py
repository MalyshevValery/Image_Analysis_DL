"""Class that encapsulates training process"""
import os
from typing import Tuple, Union

from albumentations import BasicTransform, Compose
from tensorflow.keras.models import Model

from .data import Loader, DataGenerator, AbstractStorage
from .utils import RunParams
from .utils.runparams import DEFAULT_PARAMS

Storages = Union[AbstractStorage, Tuple[AbstractStorage, ...]]


class TrainWrapper:
    """
    This class encapsulate model training process.
    It allows training, evaluation and checking validity of setup

    Training workflow
    >>> loader = Loader(...) # Loader for data
    >>> model = Model(...) # Model to train
    >>> params = RunParams(...) # Params for fit, evaluate and predict
    >>> tw = TrainWrapper(loader, model, 'job', ...)
    >>> tw.model.compile(...) # Compile model
    >>> tw.model.summary()

    >>> callbacks = [...] # Callbacks
    >>> tw.train(callbacks)
    >>> tw.evaluate('metric')


    :param loader: Loader for data
    :param model: Model
    :param job_dir: directory to save model, events, predictions
    :param train_val_test: split on training, validation and test sets
    :param batch_size: batch size
    :param augmentation_train: Augmentations for train data
        (will be merged with augmentations_all)
    :param augmentation_all: Augmentations applied to data from all sets
    :param generator_params: Parameters for training
    """

    def __init__(self, loader: Loader, model: Model, job_dir: str,
                 train_val_test=(0.8, 0.1, 0.1), batch_size=1,
                 augmentation_train: BasicTransform = None,
                 augmentation_all: BasicTransform = None,
                 generator_params: RunParams = DEFAULT_PARAMS):
        self._loader = loader
        self._train_val_test = train_val_test
        self._generator_params = generator_params
        self._model = model

        self._augmentation_train = augmentation_train
        self._augmentation_all = augmentation_all
        self._composed_augmentation = self._augmentation_train
        if self._augmentation_all and self._augmentation_train:
            self._composed_augmentation = Compose(
                [self._augmentation_train, self._augmentation_all])
        elif self._augmentation_all:
            self._composed_augmentation = self._augmentation_all

        train_keys, val_keys, test_keys = loader.split(train_val_test)
        self._train_gen = DataGenerator(train_keys, loader, batch_size,
                                        self._composed_augmentation)
        self._val_gen = DataGenerator(val_keys, loader, batch_size,
                                      shuffle=False,
                                      augmentations=self._augmentation_all)
        self._test_gen = DataGenerator(test_keys, loader, batch_size,
                                       shuffle=False,
                                       augmentations=self._augmentation_all)

        self._job_dir = job_dir
        if not os.path.exists(job_dir):
            os.makedirs(self._job_dir)
        elif not os.path.isdir(job_dir):
            raise FileExistsError(
                job_dir + ' file exists, directory cannot be created')

    def train(self, callbacks, weights_path=None):
        """
        Train process. After training best weights are restored if checkpoint
        callback with save_best was used
        """
        self._model.fit(self._train_gen, validation_data=self._val_gen,
                        callbacks=callbacks, **self._generator_params.dict)
        self._model.save(os.path.join(self._job_dir, 'model.h5'))
        if weights_path:
            self._model.load_weights(os.path.join(self._job_dir, weights_path))

    def evaluate(self, eval_metric: str) -> float:
        """
        Evaluate test set and returns value of eval_metric.

        Add '-' symbol before metric name to change its sign.
        Can be useful for hyper parameter optimization/
        """
        ret = self._model.evaluate(self._test_gen,
                                   **self._generator_params.eval_dict)
        ret_val = zip(self._model.metrics_names, ret)
        for entry in ret_val:
            filename = f'{entry[1]:.3f}_{entry[0]}'
            file = open(os.path.join(self._job_dir, filename), 'w')
            file.close()
        k = -1 if eval_metric[0] == '-' else 1
        return ret[self._model.metrics_names.index(eval_metric)] * k

    def check(self) -> None:
        """Checks if model works by training and testing on one batch"""
        batches = [self._train_gen[0], self._val_gen[0], self._test_gen[0]]
        x = [b[0] for b in batches]
        y = [b[1] for b in batches]
        self._model.train_on_batch(x[0], y[0])
        self._model.test_on_batch(x[1], y[1])
        self._model.predict_on_batch(x[2])

    def predict_save_test(self, storages: Storages) -> None:
        """Predicts test data and saves it to given storage"""
        predicted = self._model.predict(self._test_gen,
                                        **self._generator_params.eval_dict)

        if isinstance(storages, AbstractStorage):
            storages = (storages,)
        for i, storage in enumerate(storages):
            storage.save_array(self._test_gen.keys, predicted[i])

    @property
    def job_dir(self) -> str:
        """Returns path to job dir"""
        return os.path.abspath(self._job_dir)

    @property
    def model(self) -> Model:
        """Returns model of this train"""
        return self._model
