"""Utils for HoverNet"""
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import kornia

from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import watershed

from matplotlib import cm
import numpy
import cv2.cv2 as cv2


def sobel(kernel_size: int = 3) -> Tensor:
    """Returns Sobel filter in form of a Tensor"""
    assert kernel_size % 2 == 1
    s = kernel_size // 2
    k = torch.linspace(-s, s, kernel_size)
    kernel = torch.stack([k] * kernel_size)
    k[k == 0] = 1e-7
    div = torch.stack([k] * kernel_size)
    return kernel / (div ** 2 + div.T ** 2)


def batch_min_max(tensor: Tensor) -> Tensor:
    """Returns Batch normalized by min max"""
    v = tensor.view((tensor.shape[0], -1))
    min_vals = v.min(dim=1).values
    max_vals = v.max(dim=1).values
    mm_shape = (tensor.shape[0], *([1, ] * (tensor.ndim - 1)))
    min_vals = min_vals.view(mm_shape)
    max_vals = max_vals.view(mm_shape)
    return (tensor - min_vals) / (max_vals - min_vals)


def hover_to_inst(grad_gauss_filter: int = 7, grad_thresh: float = 0.4) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Returns function to transform HoverNet predictions into instance map
    :param grad_gauss_filter: Kernel size for sobel and blur
    :param grad_thresh: Threshold for energy map
    """
    assert 0 <= grad_thresh < 1
    assert grad_gauss_filter % 2 == 1

    def process(np: Tensor, hv: Tensor) -> Tensor:
        np_p = np.detach()
        h_raw = hv[:, :1].detach()
        v_raw = hv[:, 1:].detach()

        np_p[np_p >= 0.5] = 1
        np_p[np_p < 0.5] = 0

        h = batch_min_max(h_raw)
        v = batch_min_max(v_raw)

        s = sobel(grad_gauss_filter).to(np.device)
        sobel_h = torch.conv2d(h, s[None, None, ...])
        sobel_h = nn.functional.pad(sobel_h, [grad_gauss_filter // 2] * 4)
        sobel_v = torch.conv2d(v, s.T[None, None, ...])
        sobel_v = nn.functional.pad(sobel_v, [grad_gauss_filter // 2] * 4)

        sobel_h = 1 - batch_min_max(sobel_h)
        sobel_v = 1 - batch_min_max(sobel_v)

        overall = torch.max(sobel_h, sobel_v)
        overall = overall - (1 - np_p)
        overall[overall < 0] = 0

        energy = -(1.0 - overall) * np_p

        energy = kornia.filters.gaussian_blur2d(energy, (3, 3), sigma=(1, 1))
        energy = energy.cpu().numpy()
        overall = 1.0 * (overall >= grad_thresh)

        M = np_p - overall
        M[M < 0] = 0
        M = M.cpu().numpy()
        np_p = np_p.cpu().numpy()

        inst_map = []
        for i in range(np_p.shape[0]):
            M_i = binary_fill_holes(M[i][0]).astype('uint8')
            M_i = measurements.label(M_i)[0]
            w = watershed(energy[i][0], M_i, mask=np_p[i][0])
            inst_map.append(w)
        inst_map = numpy.stack(inst_map)[:, None]
        return torch.tensor(inst_map, device=np.device)

    return process


def draw_instances(canvas: torch.Tensor, instance_map: torch.Tensor, color: Sequence[float] = None) -> None:
    """Draw instances contours on image (MUTABLE)"""
    max_inst = int(instance_map.max())
    for j in range(1, max_inst):
        inst_map = instance_map == j
        ys, xs = torch.where(inst_map)
        if len(ys) == 0:
            continue
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]

        inst_map_crop = 255 * inst_map_crop.cpu().numpy().astype(numpy.uint8)
        inst_canvas_crop = inst_canvas_crop.cpu().numpy().copy()
        contours, _ = cv2.findContours(inst_map_crop, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(inst_canvas_crop, contours, -1, color, 2)
        canvas[y1:y2, x1:x2] = torch.tensor(inst_canvas_crop)
