"""Utility functions for afMLevel, including image processing and normalisation."""

import numpy as np
from scipy.ndimage import generate_binary_structure, label


def remove_small_zeros(arr, min_size=50):
    """
    Removes isolated 0s or small clusters of 0s in a binary array.

    Args:
        arr: 2D binary array (strictly 0s and 1s)
        min_size: Minimum area (in pixels) to keep a group of 0s

    Returns
    -------
        Cleaned array where small 0 regions are set to 1
    """
    # Label connected regions of 0s (8-connectivity)
    structure = generate_binary_structure(2, 2)
    labeled, num_features = label(1 - arr, structure=structure)  # Directly work with 0s

    # If no zero regions found, return original
    if num_features == 0:
        return arr.copy()

    # Count zero region sizes (label 0 is background 1s, so we skip it)
    region_sizes = np.bincount(labeled.ravel())[1:]  # Only counts for labels 1+

    # Identify small regions (faster than original list comprehension)
    small_labels = (
        np.where(region_sizes < min_size)[0] + 1
    )  # +1 because we skipped label 0

    # Create mask and clean
    cleaned = arr.copy()
    if len(small_labels) > 0:
        cleaned[np.isin(labeled, small_labels)] = 1

    return cleaned


def normalise(imarray):
    min_val = np.min(imarray)
    max_val = np.max(imarray)
    data_range = max_val - min_val

    imnorm = (imarray - min_val) / data_range
    return imnorm, min_val, data_range


def denormalise(imnorm, min_val, data_range):
    return imnorm * data_range + min_val


def linefit(imarray, polyx):
    mask = imarray > -np.inf
    if polyx > 0:
        x = np.arange(imarray.shape[1])
        y2 = np.zeros_like(imarray)

        for i in range(imarray.shape[0]):
            pos = mask[i, :] > 0
            y1 = imarray[i, pos]
            x1 = x[pos]
            p = np.polyfit(x1, y1, polyx, full=False, cov=False)
            # Evaluate polynomial
            y2[i, :] = np.polyval(p, x)
        return y2


def swap01(maskarray):
    num_zeros = np.sum(maskarray == 0)
    num_ones = np.sum(maskarray == 1)

    # Swap 0s and 1s if there are more 0s
    if num_zeros > num_ones:
        maskarray = 1 - maskarray
    return maskarray


def xyplanefit(imarray, polyx, polyy):
    mc = np.mean(imarray, axis=0)
    x = np.arange(0, len(mc), 1)
    p = np.polyfit(x, mc, polyx)
    p = np.poly1d(p)
    pvals = np.polyval(p, x)
    r = imarray - pvals[np.newaxis, :]

    mr = np.mean(r, axis=1)
    y = np.arange(0, len(mr), 1)
    p = np.polyfit(y, mr, polyy)
    p = np.poly1d(p)
    pvals = np.polyval(p, y)
    r = r - pvals[:, np.newaxis]

    return r
