"""Config for Rjabceva segmentation and classificiation"""
from datetime import datetime
from shutil import copyfile
from typing import Tuple

import albumentations as alb
import click
import cv2.cv2 as cv2
import torch
from ignite.metrics import Loss, Accuracy
from skimage.morphology import watershed, remove_small_objects

from imagedl import main_train
from imagedl.config import *
from imagedl.data import AbstractDataset
from imagedl.data.datasets import NumpyImageDataset
from imagedl.nn.losses import DiceLoss, CombinedLoss
from imagedl.nn.metrics import ConfusionMatrix, ProbConfusionMatrix, mean_iou
from imagedl.nn.models import HoverNet
from imagedl.nn.models.blocks import BNRelu, Mish
from imagedl.nn.optim import AdamP
from imagedl.utils.hover import draw_instances

GTData = Tuple[Tensor, Tuple[Tensor, ...]]
DATA_GLOB = '/data/malyshevvalery/KeyPointData/*.npy'

aug_p = 0.5
AUG = alb.Compose([
    # alb.RGBShift(p=aug_p, r_shift_limit=10, g_shift_limit=0,
    #              b_shift_limit=10),
    # alb.RandomBrightnessContrast(p=aug_p),
    # alb.Blur(p=aug_p, blur_limit=3),
    alb.Compose([
        alb.VerticalFlip(),
        alb.HorizontalFlip(),
        alb.RandomRotate90(),
        alb.Transpose(),
    ], p=1),
    alb.ShiftScaleRotate(p=aug_p),
    alb.GaussNoise(p=aug_p),
])


class RLoss(nn.Module):
    """Loss for HoverNet"""

    def __init__(self, classification: bool = False, mse_loss: bool = True):
        super().__init__()
        self.losses = {
            'SG_E': nn.BCEWithLogitsLoss() if not classification else nn.CrossEntropyLoss(),
            'SG_Dice': DiceLoss(),
        }
        self.MSE_loss = mse_loss
        if mse_loss:
            self.losses['KP_MSE'] = nn.MSELoss(),
            self.kp_loss = self.losses['KP_MSE']
        self.sg_loss = CombinedLoss(self.losses['SG_E'],
                                    self.losses['SG_Dice'])

    def forward(self, logits: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Calculate loss"""
        mask = targets[0] < 0.5
        loss = self.sg_loss(logits[0], targets[0])
        if self.MSE_loss:
            loss += 10 * self.kp_loss(RLoss.__apply_mask(logits[1], mask),
                                      RLoss.__apply_mask(targets[1], mask))
        return loss

    @staticmethod
    def __apply_mask(data, mask):
        data = data.permute(1, 0, 2, 3)
        mask = mask[:, 0]
        data[:, mask] *= 0
        return data.permute(1, 0, 2, 3)


def dataset_transform(data):
    """Transform for dataset"""
    image = data[..., :3]
    seg_norm_cancer_map = data[..., 3:]
    return image, seg_norm_cancer_map


def inst_class(seg, norm):
    values, idx = norm.max(1)
    mask = (seg[:, 0] > 0) & (values > 0.1)
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
    return torch.tensor(inst_imgs, device=seg.device), clazz


def net_transform(aug=None):
    """Create transform with augmentations"""

    def transform(data):
        """Transform with augmentations"""
        img, seg_norm_cancer_map = data

        if aug is not None:
            res = aug(image=img, mask=seg_norm_cancer_map)
            img = res['image']
            seg_norm_cancer_map = res['mask']
        img = torch.tensor(img).float()
        seg = torch.tensor(seg_norm_cancer_map[..., 0]).float()
        norm_cancer_map = torch.tensor(seg_norm_cancer_map[..., 1:]).float()

        img = img - 0.5
        return img.permute(2, 0, 1), (
            seg.unsqueeze(0), norm_cancer_map.permute(2, 0, 1)
        )

    return transform


class ImmunoHistConfig(Config):
    """Config for CAMELEON17 Segmentation"""

    def __init__(self, data_glob: Path) -> None:
        def get_wsi(s):
            return s.split('_')[0]

        dataset = NumpyImageDataset(data_glob, transform=dataset_transform,
                                    filename_transforms={'wsi': get_wsi})

        ids = dataset.filenames

        self.__data = dataset
        self.__keys = ids
        self.__groups = dataset.info['wsi']

        self.__stamp = datetime.now()
        self.__criterion = RLoss(mse_loss=False)
        self.job_dir = Path(
            f'Jobs_Temp/{self.__stamp:%Y-%m-%d-%H-%M-%S}_ImmunoCells_OnlySeg')

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
            model=HoverNet(1, bias=False),
            optimizer=lambda params: AdamP(params, lr=1e-4),
            criterion=self.__criterion,
            path_to_checkpoint=None
        )

    @property
    def train(self) -> TrainConfig:
        """Train config"""
        return TrainConfig(
            epochs=2,
            batch_size=4,
            test_batch_size=1,
            patience=500,
        )

    @property
    def test(self) -> TestConfig:
        """Test and Eval config"""

        def __select(i: int):
            def select(output):
                return output[0][i], output[1][i]

            return select

        __select_sg = __select(0)

        def __bin(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
            return (1.0 * (logits > 0)), targets

        def inst_trans(output: Tuple[Tensor, ...]):
            log_seg, log_norm = output[0]
            tar_seg, tar_norm = output[1]
            tar_inst, tar_class = inst_class(tar_seg, tar_norm)
            log_inst, log_class = inst_class(log_seg, log_norm)
            log_inst[tar_seg[:, 0] < 0.5] = 0.0
            tar_inst[tar_seg[:, 0] < 0.5] = 0.0
            pred = (log_inst[:, None],
                    log_class.unsqueeze(1))
            return pred, (tar_inst[:, None], tar_class.unsqueeze(1))

        # imi = InstanceMatchInfo(n_classes=2, output_transform=inst_trans)
        # ip = InstancePrecision(imi)
        # ir = InstanceRecall(imi)
        # f1 = F1Score(ip, ir)

        cm = ConfusionMatrix(output_transform=__select_sg)
        pcm = ProbConfusionMatrix(output_transform=__select_sg)
        metrics = {
            # 'IMI': imi,
            # 'IP': ip,
            # 'IR': ir,
            # 'F1': f1,
            # 'F1_Mean': f1.mean(),
            # 'ICM': InstanceConfusionMatrix(imi),

            'SG_accuracy': Accuracy(
                output_transform=lambda x: __bin(*__select_sg(x))),
            'SG_CM': cm,
            'SG_PCM': pcm,
            'SG_IoU_Bin': mean_iou(cm),
            'SG_IoU_Prob': mean_iou(pcm),
        }
        metric_d = {'SG': 0, 'KP': 1}
        for m in self.__criterion.losses:
            metrics[m] = Loss(self.__criterion.losses[m],
                              output_transform=__select(metric_d[m[:2]]))

        return TestConfig(metrics=metrics, eval_metric='SG_IoU_Bin',
                          # 'F1_Mean',
                          test_best_model=True,
                          train_metric_names=['SG_E', 'loss'])

    def visualize(self, inp: torch.Tensor, target: torch.Tensor,
                  pred: torch.Tensor = None) -> Dict[
        str, torch.Tensor]:
        """Visualize input and ground truth. Visualize predictions if passed"""
        images = inp.permute(0, 2, 3, 1) + 0.5
        images = images.detach()

        result = {
            'truth': images.clone(),
        }
        result['truth'][..., 0] *= (1 - 0.4 * target[0][:, 0])
        result['truth'][..., 2] *= (1 - 0.4 * target[0][:, 0])

        if pred is not None:
            seg_pred = pred[0][:, 0].detach().sigmoid()
            norm_canc_pred = pred[1].permute(0, 2, 3, 1).detach()
            norm_canc_pred[norm_canc_pred < 0] = 0.0
            norm_canc_pred /= norm_canc_pred.max()

            result['pred'] = images.clone()
            result['pred'][..., 0] *= (
                    1 - 0.3 * seg_pred.to(result['pred'].device))
            result['pred'][..., 2] *= (
                    1 - 0.3 * seg_pred.to(result['pred'].device))
            result['pred_mask'] = torch.zeros_like(images)

            pred_inst, pred_class = inst_class(pred[0].detach(),
                                               pred[1].detach())

        tar_inst, tar_class = inst_class(target[0], target[1])
        for i in range(images.shape[0]):
            draw_instances(result['truth'][i], tar_inst[i], tar_class[i], 4)
            if pred is not None:
                draw_instances(result['pred'][i], pred_inst[i], pred_class[i],
                               4)

                result['pred_mask'][i][..., 0] = norm_canc_pred[i, ..., 1]
                result['pred_mask'][i][..., 1] = seg_pred
                result['pred_mask'][i][..., 2] = norm_canc_pred[i, ..., 0]
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
            'REMOVED', 'cancer'
        ]


@click.command()
@click.option('--train', '-t', 'train_', default=0.6, help='Training size',
              type=float)
@click.option('--val', '-v', default=0.2, help='Validation size', type=float)
@click.option('--test', '-s', default=0.2, help='Test size', type=float)
@click.option('--kfold', '-k', help='Number of folds', type=int)
def main(train_: float, val: float, test: float, kfold: int = None) -> None:
    """Main program"""
    config = ImmunoHistConfig(DATA_GLOB)
    config.job_dir.mkdir(parents=True, exist_ok=True)
    copyfile(__file__, config.job_dir / 'config.py')
    main_train(config, train_, val, test, kfold)


if __name__ == '__main__':
    main()
