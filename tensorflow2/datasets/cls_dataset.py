"""
    Classification dataset routines.
"""

__all__ = ['img_normalization']

import numpy as np


def img_normalization(img,
                      mean_rgb,
                      std_rgb):
    """
    Normalization as in the ImageNet-1K validation procedure.

    Parameters
    ----------
    img : np.array
        input image.
    mean_rgb : tuple of 3 float
        Mean of RGB channels in the dataset.
    std_rgb : tuple of 3 float
        STD of RGB channels in the dataset.

    Returns
    -------
    np.array
        Output image.
    """
    # print(img.max())
    mean_rgb = np.array(mean_rgb, np.float32) * 255.0
    std_rgb = np.array(std_rgb, np.float32) * 255.0
    img = (img - mean_rgb) / std_rgb
    return img
