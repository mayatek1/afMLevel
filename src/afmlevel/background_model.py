"""
Background prediction and leveling using a trained U-Net model.

This module implements the `level_ml_bg` function, which applies a trained U-Net model
to predict the background (tilt, scan line variation and scanning artifacts) of AFM
images and level them by subtracting this background from the original data. The
function can handle both single 2D images and 3D stacks, with options for returning
either the predicted background or the levelled image. The implementation includes
efficient tiling for large images, caching of the loaded model for performance, and
handling of normalisation and denormalisation to ensure accurate predictions.

The `level_ml_bg` function is the main entry point which takes an AFM image or stack,
applies the model, and returns the processed result.

Authors
-------
Maya Tekchandani, University of Leeds (Model training and Python implementation)
Daniel E. Rollins, University of Leeds (Python implementation and optimisation)

AI Transparency Note
--------------------
AI-based tools were used in certain parts of this module for limited typing/formatting
assistance and for providing debugging, refactoring and documentation suggestions. All
code paths, algorithms, and final behaviour were reviewed and validated by the authors.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from afmlevel.unet import load_unet_model
from afmlevel.utils import denormalise, linefit, normalise, xyplanefit

# ~~~~~~~~~~~~~~~~~~~~~ MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~

UNET_CONFIG = {
    "filtersize1": 9,
    "filtersize": 9,
    "leakyrelu": False,
    "dropoutprob": 0,
}

# ~~~~~~~~~~~~~~~~ TILE SPLITTING & STICHING ~~~~~~~~~~~~~~~~~~~~~~


def _split_into_tiles(imnorm: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """
    Split a normalised image into tiles for model prediction.

    Parameters
    ----------
    imnorm : np.ndarray
        Normalised 2D image to be split into tiles.

    Returns
    -------
    Tuple[List[np.ndarray], Dict[str, int]]
      - arrays: list of (256, 256) tiles
      - meta: dict with keys needed for stitching
    """
    h, w = imnorm.shape

    # create mirrored array for general shapes
    mirror_lr = np.concatenate((imnorm, np.fliplr(imnorm)), axis=1)
    mirror_all = np.concatenate((mirror_lr, np.flipud(mirror_lr)), axis=0)

    if imnorm.shape == (256, 256):
        arrays = [imnorm]
        meta = {
            "mode": 256,
            "h": h,
            "w": w,
        }

    elif imnorm.shape == (512, 512):
        arrays = [
            imnorm[0::2, 0::2],
            imnorm[0::2, 1::2],
            imnorm[1::2, 0::2],
            imnorm[1::2, 1::2],
        ]
        meta = {"mode": 512, "h": h, "w": w}

    else:
        hr = 256 * math.ceil(h / 256)  # rounded up to nearest multiple of 256
        wr = 256 * math.ceil(w / 256)
        hj = int(hr / 256)
        wj = int(wr / 256)

        arrays = []
        for i in range(hj):
            for j in range(wj):
                array = mirror_all[i:hr:hj, j:wr:wj]
                array = array[:256, :256]
                arrays.append(array)

        meta = {"mode": 0, "h": h, "w": w, "hr": hr, "wr": wr, "hj": hj, "wj": wj}

    return arrays, meta


def _stitch_from_tiles(tiles: List[np.ndarray], meta: Dict[str, int]) -> np.ndarray:
    """
    Reverse the tiling: tile list -> stitched full-resolution background prediction.

    Combines the predicted tiles back into a single image based on the metadata from
    the splitting step.

    Parameters
    ----------
    tiles : List[np.ndarray]
        List of predicted tiles from the model.
    meta : Dict[str, int]
        Metadata from the splitting step needed to correctly stitch the tiles.

    Returns
    -------
    np.ndarray
        Stitched background prediction at the original image resolution.
    """
    mode = meta["mode"]
    h, w = meta["h"], meta["w"]

    if mode == 256:
        BGstitched = tiles[0]

    elif mode == 512:
        BGstitched = np.zeros((h, w), dtype=tiles[0].dtype)
        # Put pixels back in their original interleaved positions
        BGstitched[0::2, 0::2] = tiles[0]  # TL
        BGstitched[0::2, 1::2] = tiles[1]  # TR
        BGstitched[1::2, 0::2] = tiles[2]  # BL
        BGstitched[1::2, 1::2] = tiles[3]  # BR

    else:
        hr, wr, hj, wj = meta["hr"], meta["wr"], meta["hj"], meta["wj"]
        BGstitched = np.zeros((hr, wr), dtype=tiles[0].dtype)
        for idx, array in enumerate(tiles):
            i = idx // wj
            j = idx % wj
            BGstitched[i:hr:hj, j:wr:wj] = array
        BGstitched = BGstitched[:h, :w]  # crop back to original dims

    return BGstitched


# ~~~~~~~~~~~~~~~~~~~~~~~ PROCESSING FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~


def _predict_tiles(
    model: torch.nn.Module, arrays: List[np.ndarray], device: torch.device
) -> List[np.ndarray]:
    """
    Run model on a list of 2D tiles and return list of 2D outputs (numpy).

    Parameters
    ----------
    model : torch.nn.Module
        Loaded U-Net model for background prediction.
    arrays : List[np.ndarray]
        List of normalised 2D tiles to be predicted.
    device : torch.device
        Device to run the model on (cuda or cpu).

    Returns
    -------
    List[np.ndarray]
        List of predicted 2D tiles as numpy arrays.
    """
    preds = []
    model.eval()
    with torch.no_grad():
        for image in arrays:
            # Expect model input: [B, C, H, W] with C=1
            image_tensor = (
                torch.tensor(image, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            out = model(image_tensor)  # [1, 1, 256, 256]
            preds.append(out.squeeze(0).squeeze(0).detach().cpu().numpy())  # [256, 256]
    return preds


def _process_single_image(
    model: torch.nn.Module,
    image: np.ndarray,
    lineorder: int,
    polyx: int = 1,
    polyy: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single 2D image.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded U-Net model for background prediction.
    image : np.ndarray
        Single 2D AFM image to be processed.
    lineorder : int
        Polynomial order for the final line fit.
    polyx : int, optional
        Polynomial order for the initial plane fit in x-direction (default is 1).
    polyy : int, optional
        Polynomial order for the initial plane fit in y-direction (default is 1).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
      predictedBG_linefit (2D), predictedLev (2D)
    """
    # Plane fit and normalise (same as before)
    im_planefit = xyplanefit(image, polyx, polyy)
    imnorm, minval, datarange = normalise(im_planefit)

    # Split -> predict -> stitch
    arrays, meta = _split_into_tiles(imnorm)
    tiles_pred = _predict_tiles(model, arrays, device=next(model.parameters()).device)
    BGstitched = _stitch_from_tiles(tiles_pred, meta)

    # Denormalise and final line fit
    BG_denorm = denormalise(BGstitched, minval, datarange)
    predictedBG_linefit = linefit(BG_denorm, lineorder)

    # Leveled image
    predictedLev = im_planefit - predictedBG_linefit
    return predictedBG_linefit, predictedLev


# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~


def level_ml_bg(
    imarray: np.ndarray,
    lineorder: int,
    model_path: str,
    background: bool = False,
    zero_median: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Apply the trained U-Net model to predict background & level an AFM image or stack.

    Parameters
    ----------
    imarray : np.ndarray
        Input raw AFM image (H, W) or stack (N, H, W).
    lineorder : int
        Polynomial order for the final line fit.
    model_path : str
        Path to the trained model (.pth).
    background : bool, optional
        If True, return predicted background after the final line fit; else return
        levelled image.
    zero_median : bool, optional
        If True and input is a stack, subtract the per-slice median from each levelled
        slice.
    device : str or None, optional
        'cuda' or 'cpu'. If None, auto-selects CUDA if available.

    Returns
    -------
    np.ndarray
        If input is 2D: returns (H, W)
        If input is 3D: returns (N, H, W)
    """
    if imarray.ndim not in (2, 3):
        raise ValueError(f"imarray must be 2D or 3D, got shape {imarray.shape}")

    # Select device
    torch_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load model once
    n_channels = 1
    model = load_unet_model(model_path, n_channels, UNET_CONFIG, torch_device)

    # Process 2D
    if imarray.ndim == 2:
        predictedBG, predictedLev = _process_single_image(model, imarray, lineorder)
        return predictedBG if background else predictedLev

    # Process 3D stack
    N = imarray.shape[0]
    BG_list = []
    Lev_list = []
    for i in range(N):
        predictedBG, predictedLev = _process_single_image(model, imarray[i], lineorder)
        BG_list.append(predictedBG)
        if zero_median:
            median = np.median(predictedLev)
            predictedLev = predictedLev - median
        Lev_list.append(predictedLev)

    BG_stack = np.stack(BG_list, axis=0)
    Lev_stack = np.stack(Lev_list, axis=0)
    return BG_stack if background else Lev_stack
