"""Utils for HoverNet"""
from typing import Callable

import cv2.cv2 as cv2
import kornia
import matplotlib
import numpy
import torch
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from torch import Tensor, nn


def sobel(kernel_size: int = 3) -> Tensor:
    """Returns Sobel filter in form of a Tensor"""
    assert kernel_size % 2 == 1
    s = kernel_size // 2
    k = torch.linspace(-s, s, kernel_size)
    kernel = torch.stack([k] * kernel_size)
    k[k == 0] = 1e-7
    div = torch.stack([k] * kernel_size)
    return kernel / (div ** 2 + div.T ** 2)


def batch_min_max(tensor: Tensor, alpha: float = 0.09) -> Tensor:
    """Returns Batch normalized by min max"""
    device = tensor.device
    tensor = tensor.detach().cpu()
    mn = (tensor == 0).sum(dim=(1, 2, 3)).float()
    mn /= numpy.prod(tensor.shape[1:])
    q = alpha * (1 - mn) * 50
    n_samples = tensor.shape[0]
    add_view = [1] * (tensor.ndim - 1)
    low = torch.tensor(
        [numpy.percentile(tensor[i], q[i]) for i in range(n_samples)]).float()
    high = torch.tensor([numpy.percentile(tensor[i], 100 - q[i]) for i in
                         range(n_samples)]).float()
    low = low.view(-1, *add_view)
    high = high.view(-1, *add_view)
    return (tensor - low / (high - low)).to(device)


def hover_to_inst(grad_gauss_filter: int = 7, grad_thresh: float = 0.4) -> \
        Callable[[Tensor, Tensor], Tensor]:
    """
    Returns function to transform HoverNet predictions into instance map
    :param grad_gauss_filter: Kernel size for sobel and blur
    :param grad_thresh: Threshold for energy map
    """
    assert 0 <= grad_thresh < 1
    assert grad_gauss_filter % 2 == 1

    def process(np: Tensor, hv: Tensor) -> Tensor:
        """Process function"""
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

        m = np_p - overall
        m[m < 0] = 0
        m = m.cpu().numpy()
        np_p = np_p.cpu().numpy()

        inst_map = []
        for i in range(np_p.shape[0]):
            m_i = binary_fill_holes(m[i][0]).astype('uint8')
            m_i = remove_small_objects(m_i > 0, 10)
            m_i = measurements.label(m_i)[0]
            w = watershed(energy[i][0], m_i, mask=np_p[i][0])
            inst_map.append(w)
        inst_map = numpy.stack(inst_map)[:, None]
        return torch.tensor(inst_map, device=np.device)

    return process


colour_map = matplotlib.cm.get_cmap('jet')


def draw_instances(canvas: torch.Tensor, instance_map: torch.Tensor,
                   classes: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Draw instances contours on image"""
    max_inst = int(instance_map.max())
    for j in range(1, max_inst + 1):
        inst_map = instance_map == j
        ys, xs = torch.where(inst_map)
        clazz, counts = classes[ys, xs].unique(sorted=True, return_counts=True)
        if len(ys) == 0:
            continue
        clazz = clazz[counts.argmax()].item()
        clazz /= (n_classes - 1)
        colour = colour_map(clazz)[:-1]
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
        cv2.drawContours(inst_canvas_crop, contours, -1, colour, 1)
        canvas[y1:y2, x1:x2] = torch.tensor(inst_canvas_crop)
    return canvas


def hv_from_inst(inst_map):
    hv_grad = torch.zeros((*inst_map.shape, 2), dtype=torch.float)
    vals = torch.unique(inst_map, sorted=True)
    for i in vals.numpy():
        if i == 0:
            continue
        rr, cc = torch.where(inst_map == i)
        if cc.max() - cc.min() < 2 or rr.max() - rr.min() < 2:
            continue
        if len(rr) == 0:
            continue
        hv_grad[rr, cc, 0] = 2 * (cc.float() - cc.min()) / (
                cc.max() - cc.min()) - 1
        hv_grad[rr, cc, 1] = 2 * (rr.float() - rr.min()) / (
                rr.max() - rr.min()) - 1
    return hv_grad
