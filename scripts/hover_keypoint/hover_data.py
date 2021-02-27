"""Config for Rjabceva segmentation and classificiation"""
from datetime import datetime
from shutil import copyfile
from typing import Tuple

import albumentations as alb
import click
import cv2.cv2 as cv2
import torch
from skimage.morphology import watershed, remove_small_objects

from imagedl import main_train
from imagedl.config import *
from imagedl.data import AbstractDataset
from imagedl.data.datasets import NumpyImageDataset
from imagedl.nn.metrics.f1score import F1Score
from imagedl.nn.metrics.instance import InstanceMatchInfo, InstancePrecision, \
    InstanceRecall, InstanceConfusionMatrix
from imagedl.nn.models import HoverNet
from imagedl.nn.models.blocks import BNRelu, Mish
from imagedl.nn.optim import AdamP
from imagedl.utils.hover import draw_instances

GTData = Tuple[Tensor, Tuple[Tensor, ...]]
PREFIX = '/data/malyshevvalery/HoverNet_Data_Prepared/'
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


def get_transform(n_class):
    def dataset_transform(data):
        """Transform for dataset"""
        image = data[..., :3]
        if n_class is None:
            inst_map = data[..., 3:]
        else:
            inst_map = data[..., 3:3 + n_class]
        return image, inst_map

    return dataset_transform


def inst_class(norm):
    values, idx = norm.max(1)
    mask = values > 0.2
    clazz = (idx + 1) * mask

    mask = mask.cpu().numpy()
    values = values.cpu().numpy()
    inst_imgs = []
    for i in range(values.shape[0]):
        processed = watershed((1 - values[i]), mask=mask[i],
                              connectivity=2)
        processed = remove_small_objects(processed, 64)
        inst_imgs.append(processed)
    inst_imgs = np.stack(inst_imgs)
    return torch.tensor(inst_imgs, device=norm.device), clazz


def net_transform(aug=None):
    """Create transform with augmentations"""

    def transform(data):
        """Transform with augmentations"""
        img, inst_map = data
        if aug is not None:
            res = aug(image=img, mask=inst_map)
            img = res['image']
            inst_map = res['mask']
        img = torch.tensor(img).float()
        img = img - 0.5
        inst_map = torch.tensor(inst_map).float()
        return img.permute(2, 0, 1), inst_map.permute(2, 0, 1)

    return transform


class CellDetConfig(Config):
    """Config for CAMELEON17 Segmentation"""

    def __init__(self, dataset_name: str, job_dir: str, test=False) -> None:
        if test:
            print('Test set')
        if dataset_name == 'allcpm':
            dataset_name = 'cpm*'
        data_glob = f'{PREFIX}{dataset_name}/*.npy'

        def get_wsi(s):
            return '_'.join(s.split('_')[:-2])

        def get_fw(s):
            split = s.split('/')
            return split[-2] + '/' + get_wsi(split[-1])

        self.n_class = None if dataset_name != 'consep' else 3
        dataset = NumpyImageDataset(data_glob,
                                    transform=get_transform(self.n_class),
                                    filename_transforms={'wsi': get_wsi},
                                    path_transform={'fold_wsi': get_fw})
        if test:
            trtest = np.array([f.split('_')[0] for f in dataset.filenames])
            self.__split = Split(np.where(trtest == 'train')[0],
                                 np.where(trtest == 'test')[0], [])
            print('Split created')
        else:
            self.__split = None
        ids = dataset.filenames
        self.__data = dataset
        self.__keys = ids
        if dataset_name is not 'cpm*':
            self.__groups = dataset.info['wsi']
        else:
            self.__groups = dataset.info['fold_wsi']

        self.__stamp = datetime.now()
        self.__criterion = nn.MSELoss()
        self.job_dir = Path(
            f'Jobs/{job_dir}/{self.__stamp:%Y-%m-%d-%H-%M-%S}_{dataset_name}')

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
            split=self.__split
        )

    @property
    def model_config(self) -> ModelConfig:
        """Returns Model Config"""
        BNRelu.activation = Mish
        n = self.n_class if self.n_class is not None else 1
        return ModelConfig(
            model=HoverNet(n, remove_hv=True, increased=True, n_dense=(2, 4)),
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

        def inst_trans(output: Tuple[Tensor, ...]):
            log_norm = output[0]
            tar_norm = output[1]
            tar_inst, tar_class = inst_class(tar_norm)
            log_inst, log_class = inst_class(log_norm)
            pred = (log_inst[:, None], log_class.unsqueeze(1))
            return pred, (tar_inst[:, None], tar_class.unsqueeze(1))

        def unite_inst(output: Tuple[Tensor, ...]):
            log_norm = output[0]
            tar_norm = output[1]
            return inst_trans((log_norm.max(1)[0][:, None],
                               tar_norm.max(1)[0][:, None]))

        metrics = {}
        metrics['IMI'] = InstanceMatchInfo(self.n_class,
                                           output_transform=inst_trans)
        metrics['Prec'] = InstancePrecision(metrics['IMI'])
        metrics['Rec'] = InstanceRecall(metrics['IMI'])
        metrics['F1'] = F1Score(metrics['Prec'], metrics['Rec'])
        metrics['ICM'] = InstanceConfusionMatrix(metrics['IMI'])
        if self.n_class is not None:
            metrics['Prec_mean'] = metrics['Prec'].mean()
            metrics['Rec_mean'] = metrics['Rec'].mean()
            metrics['F1_mean'] = metrics['F1'].mean()

            metrics['IMI_un'] = InstanceMatchInfo(output_transform=unite_inst)
            metrics['Prec_un'] = InstancePrecision(metrics['IMI_un'])
            metrics['Rec_un'] = InstanceRecall(metrics['IMI_un'])
            metrics['F1_un'] = F1Score(metrics['Prec_un'], metrics['Rec_un'])
        eval_metric = 'F1' if self.n_class is None else 'F1_mean'
        return TestConfig(metrics=metrics, eval_metric=eval_metric,
                          test_best_model=True, train_metric_names=None)

    def visualize(self, inp: torch.Tensor, target: torch.Tensor,
                  pred: torch.Tensor = None) -> Dict[
        str, torch.Tensor]:
        """Visualize input and ground truth. Visualize predictions if passed"""
        images = inp.permute(0, 2, 3, 1) + 0.5
        images = images.detach()

        result = {
            'truth': images.clone(),
        }

        if pred is not None:
            norm_canc_pred = pred.permute(0, 2, 3, 1).detach()
            norm_canc_pred[norm_canc_pred < 0] = 0.0
            norm_canc_pred /= norm_canc_pred.max()

            result['pred'] = images.clone()
            result['pred_mask'] = torch.zeros_like(images)
            pred_inst, pred_class = inst_class(pred.detach())
        tar_inst, tar_class = inst_class(target)
        for i in range(images.shape[0]):
            draw_instances(result['truth'][i], tar_inst[i], tar_class[i], 4)
            if pred is not None:
                draw_instances(result['pred'][i], pred_inst[i], pred_class[i],
                               4)

                result['pred_mask'][i][..., 0] = norm_canc_pred[i, ..., 0]
        for k in result.keys():
            result[k] = (255 * result[k]).cpu().byte()
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
    test_split = (kfold is None) and (path in ['cpm17', 'consep', 'kumar'])
    config = CellDetConfig(path, jobdir, test_split)
    config.job_dir.mkdir(parents=True, exist_ok=True)
    copyfile(__file__, config.job_dir / 'config.py')
    main_train(config, train_, val, test, kfold)


if __name__ == '__main__':
    main()
