"""Class that encapsulates training process"""
import os
from typing import Tuple, Iterable, Dict

from albumentations import BasicTransform, Compose, to_dict
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model

from imports.utils.types import OneMany, to_seq
from .data import Loader, DataGenerator, AbstractStorage, AugmentationMap
from .utils import RunParams, DEFAULT_PARAMS

_TrainValTest = Tuple[float, float, float]
_Callbacks = Iterable[Callback]


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
    :param augmentation_map: Augmentation mapping to input/output
    :param generator_params: Parameters for training
    """

    def __init__(self, loader: Loader, model: Model, job_dir: str,
                 train_val_test: _TrainValTest = (0.8, 0.1, 0.1),
                 batch_size: int = 1, augmentation_train: BasicTransform = None,
                 augmentation_all: BasicTransform = None,
                 augmentation_map: AugmentationMap = None,
                 generator_params: RunParams = DEFAULT_PARAMS):
        self._loader = loader
        self._train_val_test = train_val_test
        self._generator_params = generator_params
        self._model = model
        self._batch_size = batch_size

        self._aug_train = augmentation_train
        self._aug_all = augmentation_all
        self._aug_map = augmentation_map
        self._aug_composed = self._aug_train
        if self._aug_all and self._aug_train:
            self._aug_composed = Compose([self._aug_train, self._aug_all])
        elif self._aug_all:
            self._aug_composed = self._aug_all

        train_keys, val_keys, test_keys = loader.split(train_val_test)
        self._train_gen = DataGenerator(train_keys, loader, batch_size,
                                        self._aug_map, self._aug_composed, True)
        self._val_gen = DataGenerator(val_keys, loader, batch_size,
                                      self._aug_map, self._aug_all, False)
        self._test_gen = DataGenerator(test_keys, loader, batch_size,
                                       self._aug_map, self._aug_all, False)

        self._job_dir = job_dir
        if not os.path.exists(job_dir):
            os.makedirs(self._job_dir)
        elif not os.path.isdir(job_dir):
            raise FileExistsError(
                job_dir + ' file exists, directory cannot be created')

    def train(self, callbacks: _Callbacks, weights_path: str = None) -> None:
        """
        Train process. After training best weights are restored if checkpoint
        callback with save_best was used
        """
        self._model.fit(self._train_gen, validation_data=self._val_gen,
                        callbacks=callbacks, **self._generator_params.dict)
        self._model.save(os.path.join(self._job_dir, 'model.h5'))
        if weights_path:
            self._model.load_weights(os.path.join(self._job_dir, weights_path))

    def evaluate(self, eval_metric: str = None) -> float:
        """
        Evaluate test set and returns value of eval_metric. Saves all metrics to
        job_dir

        Add '-' symbol before metric name to change its sign.
        Can be useful for hyper parameter optimization/
        """
        ret = self._model.evaluate(self._test_gen,
                                   **self._generator_params.eval_dict)
        if ret is None:
            raise ValueError('No return from evaluate')
        if isinstance(ret, float):
            gen_ret = (ret,)
        else:
            gen_ret = ret

        ret_val = zip(self._model.metrics_names, ret)
        for entry in ret_val:
            filename = f'{entry[1]:.3f}_{entry[0]}'
            open(os.path.join(self._job_dir, filename), 'w').close()

        if eval_metric is None:
            return gen_ret[0]

        k = -1 if eval_metric[0] == '-' else 1
        return gen_ret[self._model.metrics_names.index(eval_metric)] * k

    def check(self) -> None:
        """Checks if model works by training and testing on one batch"""
        batches = [self._train_gen[0], self._val_gen[0], self._test_gen[0]]
        x = [b[0] for b in batches]
        y = [b[1] for b in batches]
        self._model.train_on_batch(x[0], y[0])
        self._model.test_on_batch(x[1], y[1])
        self._model.predict_on_batch(x[2])

    def predict_save_test(self, storages: OneMany[AbstractStorage]) -> None:
        """Predicts test data and saves it to given storage"""
        predicted = self._model.predict(self._test_gen,
                                        **self._generator_params.eval_dict)
        predicted = to_seq(predicted)
        for i, storage in enumerate(to_seq(storages)):
            storage.save_array(self._test_gen.keys, predicted[i])

    @property
    def job_dir(self) -> str:
        """Returns path to job dir"""
        return os.path.abspath(self._job_dir)

    @property
    def model(self) -> Model:
        """Returns model of this train"""
        return self._model

    def to_json(self) -> Dict[str, object]:
        """Returns JSON configuration for this train wrapper"""
        aug_all = to_dict(self._aug_all) if self._aug_all else None
        aug_train = to_dict(self._aug_train) if self._aug_train else None
        aug_map = self._aug_map.to_json() if self._aug_map else None

        return {
            'loader': self._loader.to_json(),
            'job_dir': self._job_dir,
            'generator_params': self._generator_params.dict,
            'batch_size': self._batch_size,
            'train_val_test': self._train_val_test,
            'augmentation_map': aug_map,
            'augmentation_all': aug_all,
            'augmentation_train': aug_train,
            'model': self._model.to_json()
        }
