"""PANDA Test script"""
import os
from pathlib import Path
from typing import Tuple

import geffnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from torch import nn, Tensor
from tqdm import tqdm

from imagedl.nn.metrics import ROCCurve


class NetConf(nn.Module):
    """Module for predicting class + confidence"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = geffnet.efficientnet_b7(num_classes=num_classes)
        self.conf = nn.Linear(2560, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict"""
        x = self.net.features(x)
        x = self.net.global_pool(x)
        x = x.flatten(1)
        if self.net.drop_rate > 0.:
            x = geffnet.F.dropout(x, p=self.net.drop_rate,
                                  training=self.net.training)
        clazz = self.net.classifier(x)
        conf = self.conf(x)[:, 0]
        return clazz, conf


# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
NET = NetConf(5)
BATCH_SIZE = 16
CHECKPOINT = '/home/malyshevvalery/Image_Analysis_DL/Experiments/MS/2021-02-15_PAN1_Conf/best_checkpoint_AUC_Class_mean=0.9764999151229858.pth'
DEVICE = 'cuda:0'

if __name__ == '__main__':
    INPUT_DIR = Path('/data/malyshevvalery/MS_Panda/Class_LVL1_256_rad_panda')
    SAVE_TO = Path('/data/malyshevvalery/MS_Panda/Predicted_Conf1_Scores.csv')

    roc = ROCCurve(5)

    filenames = os.listdir(INPUT_DIR)
    if not SAVE_TO.exists():
        NET.load_state_dict(
            torch.load(CHECKPOINT, map_location=DEVICE)['model'])
        NET.to(DEVICE)
        NET.eval()
        scores = []
        all_classes = []
        with torch.no_grad():
            for i in tqdm(range(0, len(filenames), BATCH_SIZE)):
                batch_files = filenames[i:i + BATCH_SIZE]
                classes = torch.tensor([int(f[-5]) for f in batch_files]).to(
                    DEVICE)
                images = [imread(INPUT_DIR / file) for file in batch_files]
                tensor_images = torch.tensor(np.stack(images))
                tensor_images = (tensor_images / 255 - 0.5).float()
                tensor_images = tensor_images.permute(0, 3, 1, 2)
                tensor_images = tensor_images.to(DEVICE)
                predicted, conf = NET(tensor_images)
                scores.append(predicted.softmax(1))
                all_classes.append(classes)
        t_scores = torch.cat(scores, 0)
        t_classes = torch.cat(all_classes, 0)
        print("#Making a df")
        df = pd.DataFrame(data={
            'file': filenames,
            'classes': t_classes.cpu().numpy(),
            'norm': t_scores[:, 0].cpu().numpy(),
            'epi': t_scores[:, 1].cpu().numpy(),
            '3': t_scores[:, 2].cpu().numpy(),
            '4': t_scores[:, 3].cpu().numpy(),
            '5': t_scores[:, 4].cpu().numpy(),
        })
        print('#Saving')
        df.to_csv(str(SAVE_TO), index=False)
    else:
        print('#Loading')
        df = pd.read_csv(str(SAVE_TO))
        t_classes = torch.tensor(df['classes']).to(DEVICE)
        arr = df.loc[:, 'norm':].to_numpy()
        t_scores = torch.tensor(arr).to(DEVICE)
    roc.update((t_scores, t_classes))
    roc.visualize(roc.compute(), ['norm', 'epi', '3', '4', '5'])
    plt.show()

    # find best thresholds
    threshs = []
    for c in range(5):
        gr = t_classes == c
        sorted, idxs = t_scores[:, c].sort()
        gr = gr[idxs].cpu().numpy()
        tp = np.append((gr == 1)[::-1].cumsum()[::-1], 0)
        fp = np.append((gr == 0)[::-1].cumsum()[::-1], 0)
        fn = np.append((gr == 1).cumsum()[::-1], 0)[::-1]
        f1 = tp / (tp + 0.5 * (fp + fn))
        idx = np.argmax(f1)
        thresh = (sorted[idx] + sorted[idx - 1]) / 2
        threshs.append(thresh)
        plt.plot(sorted.cpu().numpy(), f1[:-1])
        print(f'Class {c}, F1 max {(f1.max()):.2f}, Thresh {thresh}')
    # Thresholding
    plt.show()

    t_threshs = torch.tensor(threshs, device=t_scores.device)
    # t_scores[t_scores < t_threshs[None]] = 0
    pred_classes = torch.argmax(t_scores, 1)
    for c in range(5):
        plt.hist(t_scores[:, c].cpu().numpy(), bins=100, alpha=0.5)
    plt.show()
    pred_classes = torch.argmax(t_scores, 1)

    print(f'Accuracy: {(pred_classes == t_classes).sum() / len(filenames)}')
    tp = (pred_classes == t_classes).sum()
    fp = torch.scatter_add(torch.zeros(5).to(DEVICE), 0, pred_classes,
                           torch.ones(t_classes.shape).to(DEVICE))
    fn = torch.scatter_add(torch.zeros(5).to(DEVICE), 0, t_classes,
                           torch.ones(t_classes.shape).to(DEVICE))
    f1 = tp / (tp + 0.5 * (fp + fn))
    print('F1:', f1.cpu().numpy())
    print('F1 mean:', f1.mean().item())
