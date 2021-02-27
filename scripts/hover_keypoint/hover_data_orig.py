"""Config for Rjabceva segmentation and classificiation"""
from datetime import datetime
from shutil import copyfile
from typing import Tuple

import albumentations as alb
import click
import cv2.cv2 as cv2
import torch
from ignite.metrics import Loss

from imagedl import main_train
from imagedl.config import *
from imagedl.data import AbstractDataset
from imagedl.data.datasets import NumpyImageDataset
from imagedl.nn.losses import HoverLoss
from imagedl.nn.metrics.f1score import F1Score
from imagedl.nn.metrics.instance import InstanceMatchInfo, InstancePrecision, \
    InstanceRecall, InstanceConfusionMatrix, PanopticQuality
from imagedl.nn.models import HoverNet
from imagedl.nn.models.blocks import BNRelu, Mish
from imagedl.nn.optim import AdamP
from imagedl.utils.hover import draw_instances, hv_from_inst, hover_to_inst

GTData = Tuple[Tensor, Tuple[Tensor, ...]]
PREFIX = '/data/malyshevvalery/HoverNet_Data_OrigNet/'
DATASET_NAME = None

aug_p = 0.5
AUG = alb.Compose([
    alb.Compose([
        alb.VerticalFlip(),
        alb.HorizontalFlip(),
        alb.RandomRotate90(),
        alb.Transpose(),
    ], p=1),
    alb.ShiftScaleRotate(p=aug_p),
    alb.GaussNoise(p=aug_p),
    alb.RandomCrop(80, 80, p=1)
])


def dataset_transform(data):
    """Transform for dataset"""
    image = torch.tensor((data[..., :3]).astype(np.uint8))
    inst_classes = torch.tensor(data[..., 3:].astype(np.int32))
    return image, inst_classes


def net_transform(aug=None):
    def transform(data):
        img, inst_classes = data
        img = img.numpy()
        inst_classes = inst_classes.numpy()
        if aug is not None:
            augmented = aug(image=img, mask=inst_classes)
            img = augmented['image']
            inst_classes = augmented['mask']

        img = torch.tensor(img / 255)
        inst = torch.tensor(inst_classes[..., 0]).long()
        hv = hv_from_inst(inst)
        img = img.float() - 0.5
        np = 1.0 * (inst > 0)

        target = (np[None], hv.permute(2, 0, 1))
        if inst_classes.shape[-1] == 2:
            classes = torch.tensor(inst_classes[..., 1]).long()
            target += (classes,)
        target += (inst[None],)
        return img.permute(2, 0, 1), target

    return transform


class HoverCellConfig(Config):
    """Config for CAMELEON17 Segmentation"""

    def __init__(self, dataset_name: str, job_dir: str) -> None:
        if dataset_name == 'allcpm':
            dataset_name = 'cpm*'
        data_glob = f'{PREFIX}{dataset_name}/*.npy'

        def get_wsi(s):
            return '_'.join(s.split('_')[:-2])

        def get_fw(s):
            split = s.split('/')
            return split[-2] + '/' + get_wsi(split[-1])

        self.n_class = 1 if dataset_name != 'consep' else 3
        dataset = NumpyImageDataset(data_glob,
                                    transform=dataset_transform,
                                    filename_transforms={'wsi': get_wsi},
                                    path_transform={'fold_wsi': get_fw})
        ids = dataset.filenames
        self.__data = dataset
        self.__keys = ids
        self.__pred_to_inst = hover_to_inst()
        if dataset_name != 'cpm*':
            self.__groups = dataset.info['wsi']
        else:
            self.__groups = dataset.info['fold_wsi']

        self.__stamp = datetime.now()
        self.__criterion: HoverLoss = HoverLoss(classification=self.n_class > 1)
        self.job_dir = Path(
            f'Jobs/{job_dir}/{self.__stamp:%Y-%m-%d-%H-%M-%S}'
            f'_{dataset_name}')

    @property
    def dataset(self) -> AbstractDataset:
        """Main dataset"""
        return self.__data

    @property
    def data(self) -> DataConfig:
        """Returns data configuration"""
        return DataConfig(
            dataset=self.__data,
            train_transform=net_transform(AUG),
            test_transform=net_transform(),
            groups=self.__groups,
            train_sampler_constructor=None,
            split=None
        )

    @property
    def model_config(self) -> ModelConfig:
        """Returns Model Config"""
        BNRelu.activation = Mish
        return ModelConfig(
            model=HoverNet(self.n_class, increased=True, n_dense=(2, 4),
                           add_np=True),
            optimizer=lambda params: AdamP(params, lr=1e-5),
            criterion=self.__criterion,
            path_to_checkpoint=None
        )

    @property
    def train(self) -> TrainConfig:
        """Train config"""
        return TrainConfig(
            epochs=500,
            batch_size=4,
            test_batch_size=1,
            patience=50,
        )

    @property
    def test(self) -> TestConfig:
        """Test and Eval config"""

        def __select_output(i: int):
            def __first_two(output: Tuple[Tensor, ...]) -> Tuple[
                Tensor, Tensor]:
                return output[0][i], output[1][i]

            return __first_two

        def __to_instances(data):
            logits, targets = data[0], data[1]
            pred_np = logits[0].sigmoid()
            clazz = logits[self.n_class - 1].argmax(1).unsqueeze(1)
            pred = (self.__pred_to_inst(pred_np, logits[1]), clazz)
            return pred, (targets[-1], targets[self.n_class - 1].unsqueeze(1))

        def __unite_inst(data):
            logits, targets = data[0], data[1]
            pred_np = logits[0].sigmoid()
            clazz = logits[0].argmax(1).unsqueeze(1)
            pred = (self.__pred_to_inst(pred_np, logits[1]), clazz)
            return pred, (targets[-1], targets[0].unsqueeze(1))

        clazz = None if self.n_class == 1 else self.n_class
        metrics = {}
        metrics['IMI'] = InstanceMatchInfo(clazz,
                                           output_transform=__to_instances)
        metrics['Prec'] = InstancePrecision(metrics['IMI'])
        metrics['Rec'] = InstanceRecall(metrics['IMI'])
        metrics['F1'] = F1Score(metrics['Prec'], metrics['Rec'])
        metrics['ICM'] = InstanceConfusionMatrix(metrics['IMI'])
        metrics['PQ'] = PanopticQuality(metrics['IMI'])
        if self.n_class > 1:
            metrics['Prec_mean'] = metrics['Prec'].mean()
            metrics['Rec_mean'] = metrics['Rec'].mean()
            metrics['F1_mean'] = metrics['F1'].mean()
            metrics['PQ_mean'] = metrics['PQ'].mean()

            # metrics['IMI_un'] = InstanceMatchInfo(output_transform=__unite_inst)
            # metrics['Prec_un'] = InstancePrecision(metrics['IMI_un'])
            # metrics['Rec_un'] = InstanceRecall(metrics['IMI_un'])
            # metrics['F1_un'] = F1Score(metrics['Prec_un'], metrics['Rec_un'])
            # metrics['PQ'] = PanopticQuality(metrics['IMI_un'])
        metric_d = {'NP': 0, 'HV': 1, 'NC': 2}
        train_names = []
        for m in self.__criterion.losses:
            train_names.append(m)
            metrics[m] = Loss(self.__criterion.losses[m],
                              output_transform=__select_output(metric_d[m[:2]]))
        mtr = 'F1_mean' if self.n_class > 1 else 'F1'
        return TestConfig(metrics=metrics, eval_metric=mtr,
                          test_best_model=True, train_metric_names=train_names)

    def visualize(self, inp: torch.Tensor, target: torch.Tensor,
                  pred: torch.Tensor = None, max_class: int = 1) -> Dict[
        str, torch.Tensor]:
        """Visualize input and ground truth. Visualize predictions if passed"""
        images = inp.permute(0, 2, 3, 1) + 0.5
        np = target[0]
        hv = target[1]
        if self.n_class > 1:
            nc = target[2]
        real = target[-1]
        instances = self.__pred_to_inst(np, hv).detach()

        if pred is not None:
            np_p = pred[0].sigmoid()
            hv_p = pred[1]
            instances_pred = self.__pred_to_inst(np_p, hv_p).detach()

        images_cpu = images.detach().cpu()
        result = {
            'real': images_cpu.clone(),
            'grad': images_cpu.clone()
        }
        if pred is not None:
            result['pred'] = images_cpu.clone()

        for i in range(images.shape[0]):
            tg_class = target[self.n_class - 1][i]
            if len(tg_class.shape) == 2:
                tg_class = tg_class[..., None]
            else:
                tg_class = tg_class.permute(1, 2, 0)
            draw_instances(result['real'][i], real[i][0], tg_class,
                           self.n_class)
            draw_instances(result['grad'][i], instances[i][0], tg_class,
                           self.n_class)
            if pred is not None:
                draw_instances(result['pred'][i], instances_pred[i][0],
                               pred[0][i].argmax(0), self.n_class)
        for k in result.keys():
            result[k] = (255 * result[k]).byte()
        return result

    def save_sample(self, visualized: Dict[str, Tensor], save_path: Path,
                    idx: np.ndarray = None) -> None:
        """Saves visualized sample"""
        if idx is None:
            for k in visualized:
                big_image = torch.cat([t for t in visualized[k]])
                cv2.imwrite(str(save_path) + f'_{k}.png',
                            big_image.numpy()[..., ::-1])
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            for k in visualized:
                vis = visualized[k]
                for i in range(vis.shape[0]):
                    filepath = save_path / (self.__keys[idx[i]] + f'_{k}.png')
                    cv2.imwrite(str(filepath),
                                vis[i].detach().numpy()[..., ::-1])

    @property
    def legend(self) -> List[str]:
        return [
            'none', 'cells'
        ]


@click.command()
@click.argument('path', nargs=1)
@click.argument('jobdir', nargs=1)
@click.option('--train', '-t', 'train_', default=0.6, help='Training size',
              type=float)
@click.option('--val', '-v', default=0.2, help='Validation size', type=float)
@click.option('--test', '-s', default=0.2, help='Test size', type=float)
@click.option('--kfold', '-k', help='Number of folds', type=int)
def main(path: str, jobdir: str, train_: float, val: float, test: float,
         kfold: int = None) -> None:
    """Main program"""
    config = HoverCellConfig(path, jobdir)
    config.job_dir.mkdir(parents=True, exist_ok=True)
    copyfile(__file__, config.job_dir / 'config.py')
    main_train(config, train_, val, test, kfold)


if __name__ == '__main__':
    main()
