import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, draw
from tqdm import tqdm

if __name__ == '__main__':
    DATA_DIR = Path('../../data/datasets/20200609')
    print(os.listdir(DATA_DIR))
    IMGS = glob(str(DATA_DIR / 'images_1024') + '/*.jpg')
    with open(str(DATA_DIR / 'raw_1024.json'), 'r') as f:
        annos = json.load(f)

    IMGS = np.array(IMGS)
    ids = [int(s.split('/')[-1][:-4]) for s in IMGS]

    IMGS = IMGS[np.argsort(ids)]
    ids = np.sort(ids)
    wsis = {a['id']: a['metadata']['wsi'] for a in annos['images'] if
            a['metadata']['classification_finished']}
    available_ids = np.array(list(wsis.keys()))
    print(f'IDS IMGS: {len(IMGS)}, IDS ANNO: {len(annos["images"])}')

    available_indexes = np.where(np.isin(ids, available_ids))[0]
    ids = np.sort(available_ids)
    IMGS = IMGS[available_indexes]
    wsis = np.array([wsis[k] for k in ids])
    uni, cnt = np.unique(wsis, return_counts=True)
    print(uni)
    print(f'WSIS: {len(uni)}')

    print(annos['categories'])
    cat_names = [
        'REMOVED',
        'neuroblast_nuclei_d',
        'neuroblast_nuclei_pd',
        'schwannian_nuclei',
        'mitosis_apoptosis',
        'other'
    ]
    cat_map = {
        1: 0,
        2: 5,
        3: 0,
        4: 5,
        5: 0,
        6: 5,
        7: 5,
        8: 5,
        9: 5,
        10: 4,
        11: 0,
        12: 1,
        13: 2,
        14: 2,
        15: 0,
        16: 5,
        17: 0,
        18: 3
    }

    print(pd.DataFrame({
        'original': [a['name'] for a in annos['categories']],
        'id': [a['id'] for a in annos['categories']],
        'to': [cat_names[cat_map[a['id']]] for a in annos['categories']]
    }))


    def from_orig_to_new(cat_id, metadata):
        new_cat_id = cat_map[cat_id]
        # if new_cat_id == 1:
        #     if metadata['type'] == 'differentiating':
        #         return 1
        #     elif metadata['type'] == 'poorly_differentiated':
        #         return 2
        #     else:
        #         return 0
        return new_cat_id


    def process_image(idx):
        img_id = ids[idx]
        img_annos = [a for a in annos['annotations'] if a['image_id'] == img_id]

        image = io.imread(IMGS[idx])
        h, w = image.shape[:2]

        class_px = np.zeros((h, w), dtype=np.uint16)
        hv_grad = np.zeros((h, w, 2), dtype=np.float32)
        instances = np.zeros((h, w), dtype=np.uint16)

        polygons = [np.array(anno['segmentation']) for anno in img_annos]
        draws = [draw.polygon(poly[0, 1::2], poly[0, ::2], instances.shape) for
                 poly in polygons]
        areas = np.array([len(p[0]) for p in draws])
        cat_ids = [from_orig_to_new(anno['category_id'], anno['metadata']) for
                   anno in img_annos]
        indices = np.argsort(areas)

        c = 1
        for i in indices:
            rr, cc = draws[i]
            cat_id = cat_ids[i]
            if cat_id == 0:
                continue
            cnt[cat_id - 1] += 1.0
            instances[rr, cc] = c
            c += 1
            class_px[rr, cc] = cat_id
            hv_grad[rr, cc, 0] = 2 * (cc - np.min(cc)) / (
                    np.max(cc) - np.min(cc)) - 1
            hv_grad[rr, cc, 1] = 2 * (rr - np.min(rr)) / (
                    np.max(rr) - np.min(rr)) - 1
        return image, instances, class_px, hv_grad


    DIR_SAVE = Path('HoverData')
    DIR_SAVE.mkdir(exist_ok=True)

    cnt = np.zeros(len(cat_names) - 1)
    shape = (1024, 1024, 5)
    for i in tqdm(range(len(ids))):
        image, nuc_px, class_px, hv_grad = process_image(i)
        res = np.concatenate(
            [image.astype(np.uint16), nuc_px[..., None], class_px[..., None]],
            axis=2)
        print(res.shape, res.dtype)
        np.save(str(DIR_SAVE / f'{ids[i]}-{wsis[i]}.npy'), res)
