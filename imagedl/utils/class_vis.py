import matplotlib.cm as cm
import numpy as np

CMAP = cm.get_cmap('nipy_spectral')


def generate_colours(n_classes):
    vals = np.linspace(0, 1, n_classes + 1)
    return CMAP(vals[1:])[:, :3]


def color_class_map(class_map, colors):
    coloured = class_map[..., None] * colors.to(class_map.device)
    coloured = coloured.sum(-2)
    return coloured.float()
