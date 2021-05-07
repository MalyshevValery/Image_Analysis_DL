from datetime import datetime
from shutil import copyfile
from typing import Tuple

import torch
from ignite.metrics import Accuracy, Loss
from torch.utils.data import TensorDataset

from imagedl import main_train
from imagedl.config import *
from imagedl.data.datasets import ComposeDataset
from scripts.test.test_loss import TestLoss
from scripts.test.test_model import TestModel


def generate_data(n: int, s: int) -> Tuple[Tensor, ...]:
    """Generate random data: 2 inputs and 2 classification outputs"""
    x1 = torch.rand(n, s, dtype=torch.float)
    x2 = torch.rand(n, s, dtype=torch.float)
    noise = torch.randn(n, dtype=torch.float) * 0.1
    y1 = torch.argmax(torch.stack((x1.mean(1) + 0.5 + noise,
                                   x2.mean(1) * 2)), 0).float()
    y2 = torch.argmin(torch.stack(((x1.mean(1)) ** 2,
                                   x2.mean(1) / 2 + noise)), 0).float()
    return x1, x2, y1, y2


class TestNetConfig(Config):
    def __init__(self, feats=100):
        self.feats = 100
        name = f'test_{datetime.now():%H-%M-%S}'
        super().__init__(Path(f'Jobs/{name}'), name)

    @property
    def data(self) -> DataConfig:
        """Provide random data config with no transforms"""
        tensors = generate_data(100, self.feats)
        data = [TensorDataset(t) for t in tensors]
        input_ds = TensorDataset(tensors[0], tensors[1])
        output_ds = TensorDataset(tensors[2], tensors[3])
        return DataConfig(
            dataset=ComposeDataset((input_ds, output_ds)),
            train_transform=lambda x: x,
            test_transform=lambda x: x,
            groups=None,
            train_sampler_constructor=None,
            split=None
        )

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(
            model=TestModel(self.feats),
            optimizer=torch.optim.Adam,
            criterion=TestLoss(),
            path_to_checkpoint=None
        )

    @property
    def train(self) -> TrainConfig:
        return TrainConfig(
            epochs=1,
            batch_size=16,
            test_batch_size=32,
            patience=32
        )

    @property
    def test(self) -> TestConfig:
        def __select_and_bin(i: int, bin=False):
            def select_and_bin(v):
                if bin:
                    pred = 1.0 * (v[0][i] > 0)
                else:
                    pred = v[0][i]
                return pred, v[1][i]

            return select_and_bin

        metrics = {}
        for i in range(2):
            select = __select_and_bin(i)
            metrics[f'Acc{i}'] = Accuracy(
                output_transform=__select_and_bin(0, True))
            metrics[f'Loss{i}'] = Loss(nn.BCEWithLogitsLoss(), select)
        metrics['Acc'] = (metrics[f'Acc0'] + metrics[f'Acc1']) / 2
        return TestConfig(
            metrics=metrics,
            eval_metric='-loss',
            test_best_model=True,
            train_metric_names=['Acc0', 'Acc1']
        )

    def visualize(self, inp: DataType, out: DataType,
                  pred: DataType = None) -> Tensor:
        vis = torch.stack(inp, 2)
        return {'inp': vis.unsqueeze(1)}

    def save_sample(self, visualized: Tensor, save_path: Path,
                    idx: np.ndarray = None) -> None:
        return

    @property
    def legend(self) -> List[str]:
        return ['Nothing', 'Something']


if __name__ == '__main__':
    config = TestNetConfig()
    config.job_dir.mkdir(parents=True, exist_ok=True)
    copyfile(__file__, config.job_dir / 'config.py')
    main_train(config, 0.8, 0.1, 0.1)
