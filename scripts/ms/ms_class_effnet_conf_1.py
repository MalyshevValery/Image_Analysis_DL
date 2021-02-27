"""Config for Rjabceva segmentation and classificiation"""
from datetime import datetime
from shutil import copyfile
from typing import Tuple

import albumentations as alb
import click
import cv2.cv2 as cv2
import geffnet
import pandas as pd
import torch
from ignite.metrics import Accuracy, Loss

from imagedl import main_train
from imagedl.config import *
from imagedl.data import AbstractDataset
from imagedl.data.datasets import NumpyImageDataset, FrameDataset, \
    ComposeDataset
from imagedl.nn.losses import MergedLoss
from imagedl.nn.metrics import Precision, Recall, ROCCurve, AUC
from imagedl.nn.metrics.f1score import F1Score
from imagedl.nn.optim import AdamP

GTData = Tuple[Tensor, Tuple[Tensor, ...]]
DATA_GLOB = '/data/malyshevvalery/MS_Panda/Class_LVL1_256_rad_panda/*.png'
THRESH = 0.01

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
])


def net_transform(aug=None):
    """Create transform with augmentations"""

    def transform(data):
        """Transform with augmentations"""
        img, out = data

        if aug is not None:
            res = aug(image=img)
            img = res['image']
        img = torch.tensor(img).float()
        img = img / 255 - 0.5
        return img.permute(2, 0, 1), (torch.tensor(out[0]).long(),
                                      torch.tensor(out[1]).float())

    return transform


class NetConf(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = geffnet.efficientnet_b7(num_classes=num_classes)
        self.conf = nn.Linear(2560, 1)

    def forward(self, x):
        x = self.net.features(x)
        x = self.net.global_pool(x)
        x = x.flatten(1)
        if self.net.drop_rate > 0.:
            x = geffnet.F.dropout(x, p=self.net.drop_rate,
                                  training=self.net.training)
        clazz = self.net.classifier(x)
        conf = self.conf(x)[:, 0]
        return clazz, conf


class HistClassConfig(Config):
    """Config for CAMELEON17 Segmentation"""

    def __init__(self, data_glob: str) -> None:
        def __transform(data):
            img = data[0][0]
            clazz = data[0][1]['class']
            conf = data[1][0]
            return img, (clazz, conf)

        filename_transforms = {
            'wsi': lambda filename: filename.split('_')[0],
            'class': lambda filename: int(filename.split('_')[4][0])
        }
        orig = NumpyImageDataset(data_glob,
                                 filename_transforms=filename_transforms,
                                 add_info=True)
        df = pd.read_csv(data_glob[:-6] + '.csv')
        sorter = dict(zip(orig.filenames, range(len(orig))))
        df['rank'] = df['filename'].map(sorter)
        df.sort_values('rank', inplace=True)
        add_dataset = FrameDataset(df, columns=('conf', 'conf'))
        self.__data = ComposeDataset((orig, add_dataset), transform=__transform)

        self.__keys = orig.filenames
        self.__groups = orig.info['wsi']
        self.__stamp = datetime.now()
        self.__criterion = MergedLoss(nn.CrossEntropyLoss(),
                                      nn.BCEWithLogitsLoss(), weights=[0.7,
                                                                       0.3])
        super().__init__(Path(
            f'Jobs/{self.__stamp:%Y-%m-%d-%H-%M-%S}_PAN1_Conf_0.6_lr5e5'))

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
        model = NetConf(5)
        return ModelConfig(
            model=model,
            optimizer=lambda p: AdamP(p, lr=5e-5),
            criterion=self.__criterion,
            path_to_checkpoint=None
        )

    @property
    def train(self) -> TrainConfig:
        """Train config"""
        return TrainConfig(
            epochs=70,
            batch_size=16,
            test_batch_size=4,
            patience=20,
        )

    @property
    def test(self) -> TestConfig:
        """Test and Eval config"""

        def select(i: int):
            def __select(output):
                return output[0][i], output[1][i].long()

            return __select

        s0 = select(0)
        s1 = select(1)

        def __bin(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
            return 1.0 * (logits > 0), targets

        metrics = {}
        # Confidence
        metrics['Acc_Conf'] = Accuracy(output_transform=lambda x: __bin(*s1(x)))
        metrics['Prec_Conf'] = Precision(output_transform=s1)
        metrics['Rec_Conf'] = Recall(output_transform=s1)
        metrics['F1_Conf'] = F1Score(metrics['Prec_Conf'], metrics['Rec_Conf'])
        metrics['ROC_Conf'] = ROCCurve(output_transform=s1)
        metrics['AUC_Conf'] = AUC(metrics['ROC_Conf'])

        def __bin_class(logits: Tensor,
                        targets: Tensor) -> Tuple[Tensor, Tensor]:
            return torch.softmax(logits, 1), targets

        # Class
        metrics['Acc_Class'] = Accuracy(
            output_transform=lambda x: __bin_class(*s0(x)))
        metrics['Prec_Class'] = Precision(5, output_transform=s0)
        metrics['Rec_Class'] = Recall(5, output_transform=s0)
        metrics['F1_Class'] = F1Score(metrics['Prec_Class'],
                                      metrics['Rec_Class'])
        metrics['ROC_Class'] = ROCCurve(5, output_transform=s0)
        metrics['AUC_Class'] = AUC(metrics['ROC_Class'])
        metrics['Prec_Class_mean'] = metrics['Prec_Class'].mean()
        metrics['Rec_Class_mean'] = metrics['Rec_Class'].mean()
        metrics['F1_Class_mean'] = metrics['F1_Class'].mean()
        metrics['AUC_Class_mean'] = metrics['AUC_Class'].mean()

        def __to_float(logits: Tensor,
                       targets: Tensor) -> Tuple[Tensor, Tensor]:
            return logits.float(), targets.float()

        metrics['Class_CE'] = Loss(self.__criterion.losses[0],
                                   output_transform=s0)
        metrics['Conf_BCE'] = Loss(self.__criterion.losses[1],
                                   output_transform=lambda x: __to_float(
                                       *s1(x)))
        return TestConfig(metrics=metrics, eval_metric='AUC_Class_mean',
                          test_best_model=True,
                          train_metric_names=[
                              'Acc_Conf', 'Acc_Class', 'Clacc_CE',
                              'Conf_BCE', 'Prec_Conf', 'Rec_Conf', 'F1_Conf',
                              'Prec_Class', 'Rec_Class', 'F1_Class']
                          )

    def visualize(self, inp: torch.Tensor, target: torch.Tensor,
                  pred: torch.Tensor = None) -> Dict[
        str, torch.Tensor]:
        """Visualize input and ground truth. Visualize predictions if passed"""

        def __draw_number(image: np.ndarray, text: str):
            return cv2.putText(image, text, (5, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        images = inp.permute(0, 2, 3, 1) + 0.5
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
                probs = torch.softmax(pred[0][i], 0)
                clazz = int(torch.argmax(probs).item())
                sr = f'{clazz} : {probs[clazz]:.2f} ({(pred[1][i].sigmoid()):.2f})'
                res = __draw_number(images[i].numpy().copy(), sr)
                result['pred'][i] = torch.tensor(res)
            res = __draw_number(copy, f'{target[0][i]} {target[1][i]}')
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
    config = HistClassConfig(DATA_GLOB)
    config.job_dir.mkdir(parents=True, exist_ok=True)
    copyfile(__file__, config.job_dir / 'config.py')
    main_train(config, train_, val, test, kfold)


if __name__ == '__main__':
    main()
