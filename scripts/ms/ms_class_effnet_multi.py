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
from ignite.metrics import Accuracy

from imagedl import main_train
from imagedl.config import *
from imagedl.data import AbstractDataset
from imagedl.data.datasets import NumpyImageDataset, FrameDataset, \
    ComposeDataset
from imagedl.nn.metrics import Recall, \
    Precision, AUC, ROCCurve
from imagedl.nn.metrics.f1score import F1Score
from imagedl.nn.optim import AdamP

GTData = Tuple[Tensor, Tuple[Tensor, ...]]
DATA_GLOB = '/data/malyshevvalery/MS_Panda/Class_LVL1_256_rad_panda/*.png'
THRESH = 0.01

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
        img, clazz = data

        if aug is not None:
            res = aug(image=img)
            img = res['image']
        img = torch.tensor(img).float()
        img = img / 255 - 0.5
        return img.permute(2, 0, 1), torch.tensor(clazz).float()

    return transform


class HistClassConfig(Config):
    """Config for CAMELEON17 Segmentation"""

    def __init__(self, data_glob: str) -> None:
        def __transform(data):
            return data[0], 1.0 * (data[1] > THRESH)

        filename_transforms = {
            'wsi': lambda filename: filename.split('_')[0],
        }
        orig = NumpyImageDataset(data_glob,
                                 filename_transforms=filename_transforms)
        df = pd.read_csv(data_glob[:-6] + '.csv')
        sorter = dict(zip(orig.filenames, range(len(orig))))
        df['rank'] = df['filename'].map(sorter)
        df.sort_values('rank', inplace=True)
        add_dataset = FrameDataset(df, columns=('epi', '5'))
        self.__data = ComposeDataset((orig, add_dataset), transform=__transform)

        self.__keys = orig.filenames
        self.__groups = orig.info['wsi']
        self.__stamp = datetime.now()
        self.__criterion = nn.BCEWithLogitsLoss()
        super().__init__(Path(
            f'Jobs/{self.__stamp:%Y-%m-%d-%H-%M-%S}_PAN1_Class_Multi'))

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
        model = geffnet.efficientnet_b7(num_classes=4)
        return ModelConfig(
            model=model,
            optimizer=lambda p: AdamP(p, lr=1e-4),
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

        def __select(output):
            return output[0], output[1]

        def __bin(logits: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
            return 1.0 * (logits > 0), targets

        metrics = {}
        metrics['Acc'] = Accuracy(
            output_transform=lambda x: __bin(*__select(x)))
        metrics['Prec'] = Precision(4, __select, True)
        metrics['Rec'] = Recall(4, __select, True)
        metrics['Prec_mean'] = metrics['Prec'].mean()
        metrics['Rec_mean'] = metrics['Rec'].mean()
        metrics['F1'] = F1Score(metrics['Prec'], metrics['Rec'])
        metrics['F1_mean'] = metrics['F1'].mean()
        metrics['ROC'] = ROCCurve(4, __select, True)
        metrics['AUC'] = AUC(metrics['ROC'])
        metrics['AUC_mean'] = metrics['AUC'].mean()
        return TestConfig(metrics=metrics, eval_metric='AUC_mean',
                          test_best_model=True,
                          train_metric_names=[
                              'Acc', 'Prec', 'Rec', 'Prec_mean', 'Rec_mean',
                              'F1', 'F1_mean']
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
    config = HistClassConfig(DATA_GLOB)
    config.job_dir.mkdir(parents=True, exist_ok=True)
    copyfile(__file__, config.job_dir / 'config.py')
    main_train(config, train_, val, test, kfold)


if __name__ == '__main__':
    main()
