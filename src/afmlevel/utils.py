"""Utility functions for afMLevel, including image processing and normalisation."""

import numpy as np
from scipy.ndimage import generate_binary_structure, label
import logging

logger = logging.getLogger(__name__)


def remove_small_zeros(arr, min_size=50, allow_bool=True, connectivity=2):
    """
    Removes isolated 0s or small clusters of 0s in a binary array.

    Parameters
    ----------
    arr : np.ndarray
        2D binary array (values {0,1} or boolean). Zeros are considered
        background to remove.
    min_size : int, optional
        Minimum area (in pixels) to keep a group of 0s. Zero regions with
        area < min_size will be set to 1.
    allow_bool : bool, optional
        If True, boolean arrays are accepted and handled; if False, dtype
        must be np.uint8.
    connectivity : {1,2}, optional
        1 for 4-connectivity, 2 for 8-connectivity in 2D.

    Returns
    -------
    np.ndarray
        Cleaned array (same shape and dtype as input), where small zero regions are set to 1.
    """
    if arr.ndim != 2:
        raise ValueError(
            f"remove_small_zeros() expects a 2D array, got shape {arr.shape}"
        )

    # Dtype handling
    if allow_bool and arr.dtype == np.bool_:
        work = ~arr  # invert booleans: True where zeros (background)
        dtype_msg = "bool"
    else:
        if arr.dtype != np.uint8:
            logger.error(
                f"remove_small_zeros() expected dtype uint8{ ' or bool' if allow_bool else ''}, got {arr.dtype}"
            )
            raise TypeError(
                f"Expected dtype uint8{ ' or bool' if allow_bool else ''}, got {arr.dtype}"
            )
        dtype_msg = "uint8"
        # Enforce binary content {0,1}
        u = np.unique(arr)
        if (
            u.size > 2
            or (u.size == 2 and not ((u[0] == 0) and (u[1] == 1)))
            or (u.size == 1 and u[0] not in (0, 1))
        ):
            raise ValueError(
                f"Array must be binary with values 0/1; got unique values {u}"
            )
        work = arr == 0  # boolean mask where zeros are True

    logger.debug(f"Correct dtype for remove_small_zeros ({dtype_msg})")

    # Connected components of zero regions
    if connectivity not in (1, 2):
        raise ValueError(
            "connectivity must be 1 (4-connectivity) or 2 (8-connectivity)"
        )
    structure = generate_binary_structure(2, connectivity)

    # label expects boolean/0-1 input; 'work' is boolean True where zeros
    labeled, num_features = label(work, structure=structure)

    if num_features == 0:
        return arr.copy()

    # Count region sizes (skip 0 which is background)
    region_sizes = np.bincount(labeled.ravel())[1:]  # shape: (num_features,)

    # Identify small zero regions
    if min_size <= 0:
        small_labels = np.array([], dtype=int)
    else:
        small_labels = (
            np.where(region_sizes < min_size)[0] + 1
        )  # +1 to map back to labels

    cleaned = arr.copy()

    if small_labels.size > 0:
        # Faster than np.isin for large label counts
        lut = np.zeros(num_features + 1, dtype=bool)  # include 0
        lut[small_labels] = True
        mask = lut[labeled]  # True where label is small
        # Set those zeros to ones, preserving dtype
        if cleaned.dtype == np.bool_:
            cleaned[mask] = True
        else:
            cleaned[mask] = 1

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
