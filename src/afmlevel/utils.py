"""Utility functions for afMLevel, including image processing and normalisation."""

import logging
from typing import Literal

import numpy as np
from scipy.ndimage import generate_binary_structure, label

logger = logging.getLogger(__name__)


Connectivity = Literal[1, 2]


def remove_small_zeros(
    arr: np.ndarray,
    min_size: int = 50,
    allow_bool: bool = True,
    connectivity: Connectivity = 2,
) -> np.ndarray:
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
        Cleaned array (same shape and dtype as input), where small zero
        regions are set to 1.
    """
    if arr.ndim != 2:
        raise ValueError(
            f"remove_small_zeros() expects a 2D array, got shape {arr.shape}"
        )
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")
    # Dtype handling
    if allow_bool and arr.dtype == np.bool_:
        work = ~arr  # invert booleans: True where zeros (background)
        dtype_msg = "bool"
    else:
        if arr.dtype != np.uint8:
            logger.error(
                f"remove_small_zeros() expected dtype uint8{ ' or bool' if allow_bool else ''}, got {arr.dtype}"  # noqa: E501
            )
            raise TypeError(
                f"Expected dtype uint8{ ' or bool' if allow_bool else ''}, got {arr.dtype}"  # noqa: E501
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


def normalise(imarray: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalise an array to the range [0, 1].

    The input array is linearly scaled such that its minimum value maps to 0
    and its maximum value maps to 1.

    Parameters
    ----------
    imarray : np.ndarray
        Input numeric array.

    Returns
    -------
    imnorm : np.ndarray
        Normalised array with values in the range [0, 1].
    min_val : float
        Minimum value of the input array.
    data_range : float
        Range of the input array (max - min).
    """
    min_val = np.min(imarray)
    max_val = np.max(imarray)
    data_range = max_val - min_val

    # Add guard against zero range to avoid division by zero
    if data_range == 0:
        return np.zeros_like(imarray, dtype=np.float32), min_val, 0.0

    imnorm = (imarray - min_val) / data_range
    return imnorm, min_val, data_range


def denormalise(imnorm: np.ndarray, min_val: float, data_range: float) -> np.ndarray:
    """
    Restore a normalised array to its original scale.

    Parameters
    ----------
    imnorm : np.ndarray
        Normalised array, typically produced by `normalise`.
    min_val : float
        Minimum value used during normalisation.
    data_range : float
        Data range used during normalisation.

    Returns
    -------
    np.ndarray
        Array rescaled to the original data range.
    """
    return imnorm * data_range + min_val


def linefit(imarray: np.ndarray, polyx: int) -> np.ndarray:
    """
    Fit a polynomial along each row of a 2D array.

    For each row, a polynomial of degree `polyx` is fitted to the valid
    (finite) values and evaluated across the full row.

    Parameters
    ----------
    imarray : np.ndarray
        2D input array.
    polyx : int
        Polynomial degree for the row-wise fit. Must be greater than 0.

    Returns
    -------
    np.ndarray
        2D array of the same shape containing the fitted values.

    Raises
    ------
    ValueError
        If `polyx` is not greater than 0.

    Notes
    -----
    Rows are fitted independently. All finite values are treated as valid.
    """
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
    else:
        raise ValueError("polyx must be > 0")


def swap01(maskarray: np.ndarray) -> np.ndarray:
    """
    Swap binary values if zeros are more frequent than ones.

    If the number of zeros exceeds the number of ones, the array values are
    inverted (0 ↔ 1). Otherwise, the array is returned unchanged.

    Parameters
    ----------
    maskarray : np.ndarray
        Binary array containing values 0 and 1.

    Returns
    -------
    np.ndarray
        Binary array with values possibly inverted.
    """
    num_zeros = np.sum(maskarray == 0)
    num_ones = np.sum(maskarray == 1)

    # Swap 0s and 1s if there are more 0s
    if num_zeros > num_ones:
        maskarray = 1 - maskarray
    return maskarray


def xyplanefit(imarray: np.ndarray, polyx: int, polyy: int) -> np.ndarray:
    """
    Remove a separable polynomial plane from a 2D array.

    A polynomial of degree `polyx` is fitted to the column-wise mean and
    subtracted. A polynomial of degree `polyy` is then fitted to the
    row-wise mean of the residual and subtracted.

    Parameters
    ----------
    imarray : np.ndarray
        2D input array.
    polyx : int
        Polynomial degree for fitting along the x-direction (columns).
    polyy : int
        Polynomial degree for fitting along the y-direction (rows).

    Returns
    -------
    np.ndarray
        Array with the fitted x-y polynomial plane removed.

    Notes
    -----
    This performs a separable plane removal and is not equivalent to a full
    2D polynomial surface fit.
    """
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
