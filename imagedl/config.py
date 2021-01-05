"""Config for training"""
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import NamedTuple, Sequence, Type, Dict, List, Callable, Optional

import ignite
import numpy as np
from torch import Tensor
from torch import nn
from torch.utils.data import Sampler

from imagedl.data import Split
from imagedl.data.datasets.abstract import AbstractDataset, Transform, DataType


class ModelConfig(NamedTuple):
    """Config for model, criterion and optimizer"""
    model: nn.Module
    optimizer: Type[object]
    criterion: nn.Module
    path_to_checkpoint: Path # None if you're not resuming


class DataConfig(NamedTuple):
    """Dataset, transforms and groups for split"""
    dataset: AbstractDataset
    groups: Sequence[object]
    train_transform: Transform
    test_transform: Transform
    train_sampler_constructor: Callable[[AbstractDataset], Sampler]
    split: Optional[Split]


class TrainConfig(NamedTuple):
    """Training procedure config"""
    epochs: int
    batch_size: int
    test_batch_size: int
    patience: int


class TestConfig(NamedTuple):
    """Config for metrics and testing"""
    metrics: Dict[str, ignite.metrics.Metric]
    eval_metric: str
    test_best_model: bool
    train_metric_names: Optional[List[str]]


class Config(metaclass=ABCMeta):
    """Configuration for training procedure"""

    def __init__(self, job_dir):
        self.job_dir = job_dir

    @property
    @abstractmethod
    def model_config(self) -> ModelConfig:
        """Returns model config"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def data(self) -> DataConfig:
        """Returns dataset config"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def train(self) -> TrainConfig:
        """Returns dataset config"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def test(self) -> TestConfig:
        """Returns dataset config"""
        raise NotImplementedError()

    @abstractmethod
    def visualize(self, inp: DataType, out: DataType,
                  pred: DataType = None) -> Tensor:
        """Visualize input and ground truth. Visualize predictions if passed"""
        raise NotImplementedError()

    @abstractmethod
    def save_sample(self, visualized: Tensor, save_path: Path,
                    idx: np.ndarray = None) -> None:
        """Saves visualized sample"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def job_dir(self) -> Path:
        """Returns job dir"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def legend(self) -> List[str]:
        """Returns list of classes or other legend"""
        raise NotImplementedError()
