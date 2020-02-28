"""Predictor"""
from typing import List, Tuple, Union

import numpy as np
from tensorflow.keras.models import Model

from .data import DataGenerator
from .data import Loader, AbstractStorage
from .utils import RunParams, DEFAULT_PARAMS

_ReturnType = Union[float, None, List[float], np.ndarray, List[np.ndarray]]


class InferenceWrapper:
    """
    This class provides interface for inference by known model saved in job_dir

    :param model: Model
    :param job_dir: Directory with job files
    :param generator_params: RunParams for generator
        (such as batch_size, workers etc.)
    :param batch_size: Batch size for storages
    :param summary: True to print summary
    """

    def __init__(self, model: Model, job_dir: str,
                 generator_params: RunParams = DEFAULT_PARAMS,
                 batch_size: int = 1, summary: bool = False):
        self.__model = model
        self.__job_dir = job_dir
        self.__generator_params = generator_params
        self.__batch_size = batch_size
        if summary:
            self.__model.summary()

    def __call__(self, input_: List[np.ndarray]) -> _ReturnType:
        """Single input prediction"""
        input_ = [i[np.newaxis] for i in input_]
        return self.__model.predict(input_)

    def predict_storage(self, storages_from: Tuple[AbstractStorage],
                        storages_to: Tuple[AbstractStorage]) -> None:
        """Inference for static data from storages. Prediction results are saved
            in provided storages_to"""
        loader = Loader(storages_from, tuple())
        gen = DataGenerator(loader.keys, loader, self.__batch_size,
                            shuffle=False, only_input=True)
        predictions = self.__model.predict(gen,
                                           **self.__generator_params.eval_dict)
        for i, storage in enumerate(storages_to):
            if storage is None:
                continue
            storage.save_array(loader.keys, predictions[i])

    @property
    def job_dir(self) -> str:
        """Returns job directory of Inference Wrapper"""
        return self.__job_dir

    @property
    def model(self) -> Model:
        """Returns Model used in this wrapper"""
        return self.__model
