"""Config for Rjabceva segmentation and classificiation"""
from datetime import datetime
from shutil import copyfile
from typing import Tuple

import albumentations as alb
import click
import cv2.cv2 as cv2
import geffnet
import torch
from ignite.metrics import Accuracy
from torch.utils.data import Subset

from imagedl import main_train
from imagedl.config import *
from imagedl.data import AbstractDataset
from imagedl.data.datasets import NumpyImageDataset, ComposeDataset
from imagedl.nn.metrics import ConfusionMatrix, ProbConfusionMatrix, Recall, \
    Precision, AUC, ROCCurve
from imagedl.nn.metrics.f1score import F1Score
from imagedl.nn.models.blocks import Mish
from imagedl.nn.optim import AdamP

GTData = Tuple[Tensor, Tuple[Tensor, ...]]
DATA_GLOB = '/data/malyshevvalery/MS_Panda/Class_LVL0_256_rad_panda/*.png'
FEAT_GLOB = '/data/malyshevvalery/MS_Panda/LVL1_features/*.npy'
CHECKPOINT = '/home/malyshevvalery/Image_Analysis_DL/Experiments/MS/2021-01-23_PAN0_Class/best_checkpoint_AUC_mean=0.9471410512924194.pth'

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


def net_transform(aug=None):
    """Create transform with augmentations"""

    def transform(data):
        """Transform with augmentations"""
        (img, feat), clazz = data

        if aug is not None:
            res = aug(image=img)
            img = res['image']
        img = torch.tensor(img).float()
        img = img / 255 - 0.5
        feat = torch.tensor(feat).permute(2, 0, 1)
        return (img.permute(2, 0, 1), feat), clazz

    return transform


class FeatNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.net = geffnet.efficientnet_b7(num_classes=n_class)
        self.feat_pool = nn.AdaptiveAvgPool2d(1)
        self.preprocess_feat = nn.Sequential(
            nn.Linear(2560, 2560),
            nn.BatchNorm1d(2560),
            Mish(),
        )
        self.classifier = nn.Linear(5120, n_class)

    def forward(self, *args):
        if len(args) == 2:
            img, feat = args
        else:
            img, feat = args[0]
        x = self.net.features(img)
        x = self.net.global_pool(x).flatten(1)
        xf = self.feat_pool(feat).flatten(1)
        if self.net.drop_rate > 0.:
            x = geffnet.F.dropout(x, p=self.net.drop_rate,
                                  training=self.net.training)
            xf = geffnet.F.dropout(xf, p=self.net.drop_rate,
                                   training=self.net.training)
        xf = self.preprocess_feat(xf)
        return self.classifier(torch.cat((x, xf), 1))


class HistClassConfig(Config):
    """Config for CAMELEON17 Segmentation"""

    def __init__(self, data_glob: str, feat_glob: str, checkpoint: str) -> None:
        filename_transforms = {
            'wsi': lambda filename: filename.split('_')[0],
            'class': lambda filename: int(filename.split('_')[4][0])
        }
        images = NumpyImageDataset(data_glob, add_info=True,
                                   filename_transforms=filename_transforms)
        feats = NumpyImageDataset(feat_glob, add_info=True,
                                  filename_transforms=filename_transforms)

        valid_ids = [v[:-4] + '.png' for v in feats.filenames]
        ids_feats = np.where(np.isin(images.filenames, valid_ids))[0]
        _, inv = np.unique(feats.filenames, return_inverse=True)
        _, idx = np.unique(np.array(images.filenames)[ids_feats],
                           return_index=True)

        def __transform(data):
            image = data[0][0]
            feat = data[1][0]
            clazz = data[0][1]['class']
            assert data[0][1]['class'] == data[1][1]['class']
            assert data[0][1]['wsi'] == data[1][1]['wsi']
            return (image, feat), clazz

        self.__data = ComposeDataset((Subset(images, ids_feats[idx][inv]),
                                      feats), transform=__transform)
        self.__keys = [f[:-4] for f in feats.filenames]
        self.__groups = feats.info['wsi']
        self.__stamp = datetime.now()
        self.__criterion = nn.CrossEntropyLoss()
        self.checkpoint = checkpoint
        super().__init__(Path(
            f'Jobs/{self.__stamp:%Y-%m-%d-%H-%M-%S}_Feats_Class'))

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
        model = FeatNet(5)
        model.net.load_state_dict(torch.load(self.checkpoint)['model'])
        return ModelConfig(
            model=model,
            optimizer=lambda params: AdamP(params, lr=1e-5),
            criterion=self.__criterion,
            path_to_checkpoint='/home/malyshevvalery/Image_Analysis_DL/Jobs/2021-02-02-13-52-38_Feats_Class/best_checkpoint_AUC_mean=0.9737270474433899.pth'
        )

    @property
    def train(self) -> TrainConfig:
        """Train config"""
        return TrainConfig(
            epochs=20,
            batch_size=16,
            test_batch_size=4,
            patience=5,
        )

    @property
    def test(self) -> TestConfig:
        """Test and Eval config"""

        def __select(output):
            return output[0], output[1]

        def __bin(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
            return torch.softmax(logits, 1), targets

        metrics = {}
        metrics['CM'] = ConfusionMatrix(5, __select)
        metrics['PCM'] = ProbConfusionMatrix(5, __select)
        metrics['Acc'] = Accuracy(
            output_transform=lambda x: __bin(*__select(x)))
        metrics['Prec'] = Precision(5, __select)
        metrics['Rec'] = Recall(5, __select)
        metrics['Prec_mean'] = metrics['Prec'].mean()
        metrics['Rec_mean'] = metrics['Rec'].mean()
        metrics['F1'] = F1Score(metrics['Prec'], metrics['Rec'])
        metrics['F1_mean'] = metrics['F1'].mean()
        metrics['ROC'] = ROCCurve(5, __select)
        metrics['AUC'] = AUC(metrics['ROC'])
        metrics['AUC_mean'] = metrics['AUC'].mean()
        return TestConfig(metrics=metrics, eval_metric='AUC_mean',
                          test_best_model=True,
                          train_metric_names=[
                              'Acc', 'Prec', 'Rec', 'Prec_mean', 'Rec_mean',
                              'F1', 'F1_mean'
                          ])

    def visualize(self, inp: torch.Tensor, target: torch.Tensor,
                  pred: torch.Tensor = None) -> Dict[
        str, torch.Tensor]:
        """Visualize input and ground truth. Visualize predictions if passed"""

        def __draw_number(image: np.ndarray, text: str):
            return cv2.putText(image, text, (5, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        images = inp[0].permute(0, 2, 3, 1) + 0.5
        images = images.detach()
        images = (255 * images).cpu().byte()
        result = {
            'truth': images,
        }

        if pred is not None:
            result['pred'] = images.clone()
        for i in range(images.shape[0]):
            copy = images[i].numpy().copy()
            if pred is not None:
                probs = torch.softmax(pred[i], 0)
                clazz = int(torch.argmax(probs).item())
                res = __draw_number(images[i].numpy().copy(),
                                    f'{clazz} : {probs[clazz]:.2f}')
                result['pred'][i] = torch.tensor(res)
            res = __draw_number(copy, f'{target[i]}')
            result['truth'][i] = torch.tensor(res)
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
            return

    @property
    def legend(self) -> List[str]:
        return [
            'norm_tissue', 'norm_epi', 'cancer_3', 'cancer_4', 'cancer_5'
        ]


@click.command()
@click.option('--train', '-t', 'train_', default=0.8, help='Training size',
              type=float)
@click.option('--val', '-v', default=0.1, help='Validation size', type=float)
@click.option('--test', '-s', default=0.1, help='Test size', type=float)
@click.option('--kfold', '-k', help='Number of folds', type=int)
def main(train_: float, val: float, test: float, kfold: int = None) -> None:
    """Main program"""
    config = HistClassConfig(DATA_GLOB, FEAT_GLOB, CHECKPOINT)
    config.job_dir.mkdir(parents=True, exist_ok=True)
    copyfile(__file__, config.job_dir / 'config.py')
    main_train(config, train_, val, test, kfold)


if __name__ == '__main__':
    main()
