"""PANDA Test script"""
from pathlib import Path
from typing import Tuple

import geffnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from openslide import OpenSlide
from torch import nn, Tensor
from tqdm import tqdm

from roll import roll


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
LEVEL = 1
TILE_SIZE = 256
STEP = 128
NET = NetConf(5)
CHECKPOINT = '/home/malyshevvalery/MS/2021-02-15_PAN1_Conf/best_checkpoint_AUC_Class_mean=0.9764999151229858.pth'
DEVICE = 'cuda:1'


def true(image: OpenSlide, mask: OpenSlide,
         level: int, step: int) -> np.ndarray:
    """Prepare true mask"""
    dim = image.level_dimensions[level]  # x y
    ds = int(image.level_downsamples[level])
    ts_l = int(TILE_SIZE * ds / mask.level_downsamples[2])
    st_l = int(step * ds / mask.level_downsamples[2])
    mask_thumb = np.array(mask.read_region((0, 0), 2, mask.level_dimensions[2]))
    mask_thumb = mask_thumb[..., 0]
    plt.imshow(mask_thumb, cmap=cmap)
    plt.show()
    excluded = roll(mask_thumb == 0, ts_l, st_l).mean(axis=(2, 3))
    result = np.zeros(excluded.shape, dtype=np.uint8)
    rows, cols = np.where(excluded < 0.3)

    for rr, cc in tqdm(zip(rows, cols), total=len(rows)):
        # Read data from a proper region
        x = cc * step * ds
        y = rr * step * ds
        tile_region = image.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE))
        tile = np.array(tile_region)[..., :3]
        mask_region = mask.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE))
        mask_tile = np.array(mask_region)[..., 0]
        # Check if the tile is background
        white_dist = np.sqrt(((1 - tile / 255) ** 2).sum(2))
        white_score = (white_dist < 0.3).mean()
        if (mask_tile == 0).mean() >= 0.3 or white_score > 0.3:
            continue
        unq, cnt = np.unique(mask_tile, return_counts=True)
        cnt = cnt / TILE_SIZE / TILE_SIZE
        if unq[0] == 0:
            cnt = cnt[1:]
            unq = unq[1:]
        well_presented = unq[cnt > 0.05]
        result[rr, cc] = np.max(well_presented)
    mask.close()
    return result


def predicted(image: OpenSlide, level: int, step: int) -> np.ndarray:
    """Predict resulting mask"""
    dim = image.level_dimensions[level]  # x y
    ds = int(image.level_downsamples[level])
    ts_l = int(TILE_SIZE * ds / image.level_downsamples[2])
    st_l = int(step * ds / image.level_downsamples[2])
    image_thumb = image.read_region((0, 0), 2, image.level_dimensions[2])
    image_thumb = np.array(image_thumb)[..., :3]
    white_dist = np.sqrt(np.sum((image_thumb / 255) ** 2, axis=2)) > 1.5
    excluded = roll(white_dist, ts_l, st_l).mean(axis=(2, 3))
    result = np.zeros(excluded.shape + (5,), dtype=np.uint8)
    rows, cols = np.where(excluded < 0.3)

    coords = []
    tiles = []
    for rr, cc in tqdm(zip(rows, cols), total=len(rows)):
        # Read data from a proper region
        x = cc * step * ds
        y = rr * step * ds
        tile_region = slide.read_region((x, y), level, (TILE_SIZE, TILE_SIZE))
        tile = np.array(tile_region)[..., :3]
        # Check if the tile is background
        white_dist = np.sqrt(((1 - tile / 255) ** 2).sum(2))
        if (white_dist < 0.3).mean() > 0.3:
            continue
        coords.append([rr, cc])
        tiles.append(tile)
    coords_arr = np.array(coords)
    tiles_arr = np.stack(tiles)
    input_data = torch.tensor(tiles_arr / 255 - 0.5).float().permute(0, 3, 1, 2)
    classes = []
    for bi in tqdm(range(0, input_data.shape[0], 4)):
        logits, conf_logits = NET(input_data[bi:bi + 4].to(DEVICE))
        clazz = torch.softmax(logits, 1).detach().cpu().numpy()
        conf = torch.sigmoid(conf_logits).cpu()
        classes.append(clazz)
    class_arr = np.concatenate(classes)
    result[coords_arr[:, 0], coords_arr[:, 1]] = class_arr
    return result


if __name__ == '__main__':
    INPUT_DIR = Path('/hdd_barracuda2/PANDA/PANDA_raw')
    IMG_DIR = INPUT_DIR / 'train_images'
    MASK_DIR = INPUT_DIR / 'train_label_masks'
    NET.load_state_dict(torch.load(CHECKPOINT)['model'])
    NET.to(DEVICE)
    NET.eval()
    cmap = ListedColormap(['black', 'cyan', 'blue', 'green', 'yellow', 'red'])

    df = pd.read_csv(INPUT_DIR / 'train.csv')
    suspicious = pd.read_csv(
        '/hdd_barracuda2/PANDA/PANDA_Suspicious_Slides.csv')
    df = df.loc[~df['image_id'].isin(suspicious['image_id'])]
    df.loc[df['gleason_score'] == 'negative', 'gleason_score'] = '0+0'
    df['gleason1'] = df['gleason_score'].str[0].astype(int)
    df['gleason2'] = df['gleason_score'].str[2].astype(int)
    df_rad = df[df['data_provider'] == 'radboud']
    df_rad = df_rad[df_rad['isup_grade'] > 0]
    print(df_rad.head().T)

    i, row = next(df_rad.iterrows())
    image_id = row['image_id']
    slide = OpenSlide(str(IMG_DIR / (image_id + '.tiff')))
    slide_mask = OpenSlide(str(MASK_DIR / (image_id + '_mask.tiff')))

    true_mask = true(slide, slide_mask, 0, TILE_SIZE)
    predicted_mask = predicted(slide, 1, STEP)
    print(true_mask.shape, predicted_mask.shape)
    slide.close()

    plt.figure(figsize=(10, 20))
    plt.subplot(121)
    plt.imshow(true_mask / 5, cmap=cmap)
    plt.subplot(122)
    # plt.imshow(predicted_mask / 5, cmap=cmap)
    plt.show()

    # prepare empty mask
    # predict on it
    # measure time
    # measure score
    # TODO: level 0 always but we can predict 1

# (repeated[1:-1:2] + repeated[2:-1:2])/2).shape
# Out[5]: (47, 20, 5
# r1 = np.concatenate([repeated[:1],(repeated[1:-1:2] + repeated[2:-1:2])/2,repeated[-2:]])
# r1.shape
# Out[7]: (50, 20, 5)
# r1 = np.concatenate([repeated[:1],(repeated[1:-1:2] + repeated[2:-1:2])/2,
#                      repeated[-1:]])
# r1.shape
# Out[9]: (49, 20, 5)
# r2 = np.concatenate([r1[:,0],(r1[:,1:-1:2] + r1[:,2:-1:2])/2,
#                      r1[:,-1:]],axis=1)
# Traceback (most recent call last):
#   File "/home/malyshevvalery/.pyenv/versions/imagedl/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3427, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-10-2d079e572222>", line 1, in <module>
#     r2 = np.concatenate([r1[:,0],(r1[:,1:-1:2] + r1[:,2:-1:2])/2,
#   File "<__array_function__ internals>", line 5, in concatenate
# ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 3 dimension(s)
# r2 = np.concatenate([r1[:,:1],(r1[:,1:-1:2] + r1[:,2:-1:2])/2,
#                      r1[:,-1:]],axis=1)
# r2.shape
# Out[12]: (49, 11, 5)
# final = r2.repeat(2,0).repeat(2,1)
# plt.imshow(final)
# Traceback (most recent call last):
#   File "/home/malyshevvalery/.pyenv/versions/imagedl/lib/python3.9/site-packages/IPython/core/interactiveshell.py", line 3427, in run_code
#     exec(code_obj, self.user_global_ns, self.user_ns)
#   File "<ipython-input-14-72cd99784291>", line 1, in <module>
#     plt.imshow(final)
#   File "/home/malyshevvalery/.pyenv/versions/imagedl/lib/python3.9/site-packages/matplotlib/pyplot.py", line 2724, in imshow
#     __ret = gca().imshow(
#   File "/home/malyshevvalery/.pyenv/versions/imagedl/lib/python3.9/site-packages/matplotlib/__init__.py", line 1447, in inner
#     return func(ax, *map(sanitize_sequence, args), **kwargs)
#   File "/home/malyshevvalery/.pyenv/versions/imagedl/lib/python3.9/site-packages/matplotlib/axes/_axes.py", line 5523, in imshow
#     im.set_data(X)
#   File "/home/malyshevvalery/.pyenv/versions/imagedl/lib/python3.9/site-packages/matplotlib/image.py", line 711, in set_data
#     raise TypeError("Invalid shape {} for image data"
# TypeError: Invalid shape (98, 22, 5) for image data
# finall = final.argmax(2) + 1
# plt.imshow(finall, cmap=cmap)
# Out[16]: <matplotlib.image.AxesImage at 0x7fa772613ac0>
# plt.show()
