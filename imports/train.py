"""Class that encapsulates training process"""
import os
from typing import Tuple, Iterable, Dict, Optional

import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model

from imports.utils.types import to_seq
from .data import Loader, StorageType, Split, DataGenerator
from .utils import RunParams, DEFAULT_PARAMS

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
    :param split: Split with train, val, test DataGenerators
    :param generator_params: Parameters for training
    """

    def __init__(self, loader: Loader, model: Model, job_dir: str,
                 split: Split, generator_params: RunParams = DEFAULT_PARAMS):
        self._loader = loader
        self._generator_params = generator_params
        self._model = model
        self._split = split

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
        self._model.fit(self._split.train, validation_data=self._split.val,
                        callbacks=callbacks, **self._generator_params.dict)
        self._model.save(os.path.join(self._job_dir, 'model.h5'))
        if weights_path:
            self._model.load_weights(weights_path)

    def evaluate(self, eval_metric: str = None) -> float:
        """
        Evaluate test set and returns value of eval_metric. Saves all metrics to
        job_dir

        Add '-' symbol before metric name to change its sign.
        Can be useful for hyper parameter optimization/
        """
        ret = self._model.evaluate(self._split.test,
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
        if eval_metric[0] == '-':
            eval_metric = eval_metric[1:]
        return gen_ret[self._model.metrics_names.index(eval_metric)] * k

    def check(self) -> None:
        """Checks if model works by training and testing on one batch"""
        batches = [self._split.train[0],
                   self._split.val[0],
                   self._split.test[0]]
        x = [b[0] for b in batches]
        y = [b[1] for b in batches]
        self._model.train_on_batch(x[0], y[0])
        self._model.test_on_batch(x[1], y[1])
        self._model.predict_on_batch(x[2])

    def predict(self, data: DataGenerator,
                storage: StorageType = None) -> Optional[np.ndarray]:
        """
        Predicts given data and saves it to the storage

        :param data: Generator of input data
        :param storage: List of storage to save predicted values,
            None to return numpy array
        """
        predicted = self._model.predict(data,
                                        **self._generator_params.eval_dict)
        if storage is None:
            return predicted
        else:
            predicted = to_seq(predicted)
            for i, storage in enumerate(to_seq(storage)):
                storage.save_array(data.keys, predicted[i])
            return None

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
        return {
            'loader': self._loader.to_json(),
            'job_dir': self._job_dir,
            'generator_params': self._generator_params.dict,
            'split': self._split.to_json(),
            'model': self._model.to_json()
        }
