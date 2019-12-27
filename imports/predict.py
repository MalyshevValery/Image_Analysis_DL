"""Predictor"""
import copy
import os

from albumentations import BasicTransform
from tensorflow.keras.models import Model

from .data import Loader, OnlineGenerator, AbstractStorage
from .data.generators import ImageGenerator
from .jsonserializable import JSONSerializable
from .models import ModelsFactory
from .wrappers import AlbumentationsWrapper


class PredictWrapper(JSONSerializable):
    """This class provides interface for inference by known model saved in job_dir"""

    def __init__(self, loader: Loader, model: Model, job_dir, augmentation: BasicTransform = None,
                 generator_params=None, batch_size=1):
        """Constructor

        :param loader: data loader
        :param model: model
        :param job_dir: directory with job
        :param augmentation: augmentation for all data
        :param generator_params: parameters from Model.predict_generator (such as batch_size, workers etc.)
        """
        self._loader = loader
        self._model = model
        self._generator = OnlineGenerator(augmentation)
        self._augmentation = augmentation
        self._job_dir = job_dir
        self._generator_params = generator_params
        self._batch_size = batch_size
        self._model.summary()

    def __call__(self, image):
        """Single input prediction"""
        image = self._loader.process_image(image)
        image = self._generator.process_image(image)
        return self._model.predict(image)

    def to_json(self):
        """Saves prediction config to dict"""
        config = {
            'loader': self._loader.to_json(),
            'model': ModelsFactory.to_json(self._model),
            'job_dir': os.path.abspath(self._job_dir),
            'batch_size': self._batch_size
        }
        if self._augmentation:
            config['augmentation_all'] = AlbumentationsWrapper.to_json(self._augmentation),
        return config

    def predict_storage(self, storage_from: AbstractStorage, storage_to: AbstractStorage):
        """Inference for static data from storage. Prediction results are saved in provided storage_to"""
        loader = self._loader.copy_for_storage(storage_from)
        gen = ImageGenerator(loader.get_keys(), loader, self._batch_size, self._augmentation, shuffle=False)
        predictions = self._model.predict_generator(gen, **self._generator_params)
        loader.save_predicted(loader.get_keys(), predictions, storage_to)

    @staticmethod
    def from_json(json, job_dir):
        """Creates Predictor from JSON config (either training or prediction configs"""
        loader = Loader.from_json(json['loader'], predict=True)
        model = ModelsFactory.load(job_dir)

        if 'augmentation_all' in json:
            augmentation = AlbumentationsWrapper.from_json(json['augmentation_all'])
        else:
            augmentation = None

        batch_size = json.get('batch_size', 1)
        gen_params = copy.deepcopy(json.get('generator_params', {}))
        gen_params.pop('epochs', None)
        return PredictWrapper(loader, model, job_dir, augmentation=augmentation, generator_params=gen_params,
                              batch_size=batch_size)
