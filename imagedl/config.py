"""Config for training"""
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import NamedTuple, Sequence, Type, Dict

import ignite
import numpy as np
from torch import Tensor
from torch import nn

from imagedl.data.datasets.abstract import AbstractDataset, Transform


class ModelConfig(NamedTuple):
    """Config for model, criterion and optimizer"""
    model: nn.Module
    optimizer: Type[object]
    criterion: nn.Module


class DataConfig(NamedTuple):
    """Dataset, transforms and groups for split"""
    dataset: AbstractDataset
    groups: Sequence[object]
    train_transform: Transform
    test_transform: Transform
    job_dir: Path


class TrainConfig(NamedTuple):
    """Training procedure config"""
    epochs: int
    batch_size: int
    patience: int


class TestConfig(NamedTuple):
    """Config for metrics and testing"""
    metrics: Dict[str, ignite.metrics.Metric]
    eval_metric: str


class Config(metaclass=ABCMeta):
    """Configuration for training procedure"""

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
    def show_samples(self, ds: AbstractDataset, job_dir: Path) -> None:
        """
        Plots data from ds and saves it to job_dir
        :param ds: Dataset to get data from
        :param job_dir: Directory to save data to
        """
        raise NotImplementedError()

    @abstractmethod
    def save_result(self, idx: np.ndarray, inp: Tensor, target: Tensor,
                    pred: Tensor, job_dir: Path) -> None:
        """
        Saves inputs, ground truth and predictions to job_dir
        :param idx: Indexes of predicted samples
        :param inp: Input data
        :param target: Ground Truth
        :param pred: Predictions
        :param job_dir: Job Directory
        """
        raise NotImplementedError()
