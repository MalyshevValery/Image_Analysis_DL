"""Utils for visualizing"""
from typing import Any

import matplotlib.cm as cm
import numpy as np

COLOR_MAP = cm.get_cmap('jet')


def generate_colours(n_classes: int) -> Any:
    """Generate colour map for specific number of classes"""
    values = np.linspace(0, 1, n_classes + 1)
    return COLOR_MAP(values[1:])[:, :3]
