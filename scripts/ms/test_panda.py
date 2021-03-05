"""PANDA Test script"""
from pathlib import Path
from time import time
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
LEVEL = 0
TILE_SIZE = 256
STEP = 256
NET = geffnet.efficientnet_b7(num_classes=5)
CHECKPOINT = '/home/malyshevvalery/MS/2021-01-23_PAN0_Class/best_checkpoint_AUC_mean=0.9471410512924194.pth'
DEVICE = 'cuda:1'


def agg_count(arr: np.ndarray, n: int = 6) -> np.ndarray:
    """Count aggregation"""
    res = np.zeros(n)
    unq, cnt = np.unique(arr, return_counts=True)
    res[unq] = cnt
    return res


def true(image: OpenSlide, mask: OpenSlide,
         level: int, step: int) -> np.ndarray:
    """Prepare true mask"""
    dim = image.level_dimensions[level]  # x y
    ds = int(image.level_downsamples[level])
    ts_l = int(TILE_SIZE * ds / mask.level_downsamples[2])
    st_l = int(step * ds / mask.level_downsamples[2])
    mask_thumb = np.array(mask.read_region((0, 0), 2, mask.level_dimensions[2]))
    mask_thumb = mask_thumb[..., 0]
    # plt.imshow(mask_thumb, cmap=cmap)
    # plt.show()
    excluded = roll(mask_thumb == 0, ts_l, st_l).mean(axis=(2, 3))
    result = np.zeros(excluded.shape, dtype=np.uint8)
    rows, cols = np.where(excluded < 0.3)

    for rr, cc in zip(rows, cols):
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
    result = np.zeros(excluded.shape + (5,), dtype=np.float32)
    rows, cols = np.where(excluded < 0.3)

    coords = []
    tiles = []
    for rr, cc in zip(rows, cols):
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
    for bi in range(0, input_data.shape[0], 4):
        logits = NET(input_data[bi:bi + 4].to(DEVICE))
        # logits, conf_logits = NET(input_data[bi:bi + 4].to(DEVICE))
        clazz = torch.softmax(logits, 1).detach().cpu().numpy()
        # conf = torch.sigmoid(conf_logits).cpu()
        classes.append(clazz)
    class_arr = np.concatenate(classes)
    result[coords_arr[:, 0], coords_arr[:, 1]] = class_arr
    return result


if __name__ == '__main__':
    INPUT_DIR = Path('/hdd_barracuda2/PANDA/PANDA_raw')
    IMG_DIR = INPUT_DIR / 'train_images'
    MASK_DIR = INPUT_DIR / 'train_label_masks'
    NET.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE)['model'])
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

    f1_names = ['f1_norm', 'f1_epi', 'f1_gl3', 'f1_gl4', 'f1_gl5']
    res_df = pd.DataFrame(columns=['id', 'acc', 'time'] + f1_names,
                          dtype=np.float32)
    res_df['id'] = res_df['id'].astype(str)
    for i, row in tqdm(df_rad.iterrows(), total=len(df_rad)):
        image_id = row['image_id']
        slide = OpenSlide(str(IMG_DIR / (image_id + '.tiff')))
        slide_mask = OpenSlide(str(MASK_DIR / (image_id + '_mask.tiff')))

        true_mask = true(slide, slide_mask, 0, TILE_SIZE)
        # ---------------------
        try:
            st = time()
            pred = predicted(slide, LEVEL, STEP)
            slide.close()
            #
            # pred = pred.repeat(2, 0).repeat(2, 1)
            # pred = np.concatenate([pred[:1], (pred[1:-1:2] + pred[2:-1:2]) / 2,
            #                        pred[-1:]])
            # pred = np.concatenate(
            #     [pred[:, :1], (pred[:, 1:-1:2] + pred[:, 2:-1:2]) / 2,
            #      pred[:, -1:]], axis=1)
            # pred = pred.repeat(2, 0).repeat(2, 1)
            pred_classes = np.argmax(pred, 2) + 1
            pred_classes[pred.sum(2) == 0] = 0
            diff = time() - st
        except Exception as e:
            print(e)
        # -------------------------

        plt.figure(figsize=(10, 20))
        plt.autoscale(False)
        plt.subplot(121)
        plt.imshow(true_mask, cmap=cmap, vmin=0, vmax=5)
        plt.subplot(122)
        plt.imshow(pred_classes, cmap=cmap, vmin=0, vmax=5)
        plt.show()

        true_mask = true_mask[:pred_classes.shape[0], :pred_classes.shape[1]]
        trues = true_mask.ravel()
        predictions = pred_classes.ravel()
        tp = agg_count(trues[trues == predictions])
        p = agg_count(predictions)
        t = agg_count(trues)
        print(p, t)
        prec = tp / p
        rec = tp / t
        acc = tp.sum() / p.sum()
        f1 = 2 * (prec * rec) / (prec + rec)
        to_add = {k: v for k, v in zip(f1_names, f1)}
        to_add['id'] = image_id
        to_add['acc'] = acc
        to_add['time'] = diff
        res_df = res_df.append(to_add, ignore_index=True)
        break
    # res_df.to_csv(f'Slide_Prediction_Results_Lvl{LEVEL}.csv', index=None)
