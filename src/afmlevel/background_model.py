"""
Background prediction and leveling using a trained U-Net model.

This module implements machine-learning-based background prediction and automated
levelling routines for atomic force microscopy (AFM) images. A trained U-Net
convolutional neural network is used to predict the large-scale, smoothly varying
background components of AFM data, including sample tilt, scan-line variation, and
instrument-related scanning artefacts. The predicted background is then subtracted
from the original data to produce a levelled image.

The core public function is:

* `level_ml_bg()` - Main entry point for users. Applies the trained U-Net model to an
  AFM image or image stack to predict the underlying background signal and returns
  either the predicted background or the levelled result (original image minus
  predicted background). The function supports both single 2D images and 3D stacks.

The implementation includes efficient tiling strategies for large images, caching of
the loaded model to avoid repeated initialisation, and careful handling of
normalisation and denormalisation to ensure numerical stability and accurate
background prediction across varying height scales.

The `level_ml_bg()` function orchestrates the full pipeline: preprocessing and
normalisation of the input data, model inference to predict the background, optional
reconstruction from tiled predictions, and subtraction of the background to generate
a levelled AFM image or stack.

Model loading and distribution
------------------------------
The trained U-Net model weights can be supplied either as a local file path or via
a Hugging Face repository identifier. By default, the model is loaded using a
Hugging Face reference of the form:

    "Heath-AFM-Lab/afMLevel-bg-unet::bg_unet.pth"

When a Hugging Face identifier is used, the model weights are downloaded on demand
(if not already cached locally) using the Hugging Face Hub and reused across calls.
This allows reproducible access to a fixed, versioned background prediction model
without bundling large binary weight files directly in the source repository.

Authors
-------
Maya Tekchandani, University of Leeds (Model training and Python implementation)
Daniel E. Rollins, University of Leeds (Python implementation and optimisation)

AI Transparency Note
--------------------
AI-based tools were used in limited parts of this module for typing, formatting, and
documentation assistance, as well as for debugging and refactoring suggestions. All
code paths, algorithms, and final behaviour were reviewed and validated by the
authors.
"""

import logging
import math

import numpy as np
import torch

from afmlevel.types import UNetConfig
from afmlevel.unet import load_unet_model
from afmlevel.utils import denormalise, linefit, normalise, xyplanefit

logger = logging.getLogger(__name__)
# ~~~~~~~~~~~~~~~~~~~~~ MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~


UNET_CONFIG: UNetConfig = {
    "filtersize": 9,
    "leakyrelu": False,
    "dropoutprob": 0,
}

# ~~~~~~~~~~~~~~~~ TILE SPLITTING & STICHING ~~~~~~~~~~~~~~~~~~~~~~


def _split_into_tiles(imnorm: np.ndarray) -> tuple[list[np.ndarray], dict[str, int]]:
    """
    Split a normalized AFM image into 256x256 tiles suitable for U-Net inference.

    This function supports three tiling modes depending on the input size:

    1. Exact 256x256 images
       - Returned as a single tile.
       - No padding or stitching needed.
       mode = 256

    2. Exact 512x512 images
       - Split into four *interleaved* 256x256 tiles using a checkerboard
         subsampling pattern:
             TL = im[0::2, 0::2]
             TR = im[0::2, 1::2]
             BL = im[1::2, 0::2]
             BR = im[1::2, 1::2]
       - Ensures every pixel appears in exactly one tile.
       - Avoids seams and preserves global continuity.
       mode = 512

    3. General case (arbitrary HxW)
       - Image is mirrored left-right and top-bottom to create a seamless padded
         canvas.
       - Padded dimensions are rounded up to the next multiples of 256.
       - Tiles are extracted from the mirrored canvas as a regular grid of
         non-overlapping 256x256 patches.
       - This avoids hard padding boundaries and reduces edge artifacts.
       mode = 0

    Parameters
    ----------
    imnorm : np.ndarray
        Normalized AFM heightmap, shape (H, W), float32 in [0, 1].

    Returns
    -------
    tiles : list of np.ndarray
        List of 256x256 tiles to be fed into the U-Net.
    meta : dict[str, int]
        Metadata required to stitch tile predictions back into the full image:
        - "mode": tiling mode (256, 512, or 0 for general)
        - "h", "w": original image size
        - "hr", "wr": padded dimensions (general mode only)
        - "hj", "wj": number of tiles vertically/horizontally (general mode only)
    """
    h, w = imnorm.shape

    # create mirrored array for general shapes
    mirror_lr = np.concatenate((imnorm, np.fliplr(imnorm)), axis=1)
    mirror_all = np.concatenate((mirror_lr, np.flipud(mirror_lr)), axis=0)

    if imnorm.shape == (256, 256):
        logger.debug("Tiling mode=256 (single tile)")
        arrays = [imnorm]
        meta = {
            "mode": 256,
            "h": h,
            "w": w,
        }

    elif imnorm.shape == (512, 512):
        logger.debug("Tiling mode=512 (4 interleaved tiles)")
        arrays = [
            imnorm[0::2, 0::2],
            imnorm[0::2, 1::2],
            imnorm[1::2, 0::2],
            imnorm[1::2, 1::2],
        ]
        meta = {"mode": 512, "h": h, "w": w}

    else:
        logger.debug("Tiling mode=general (mirror-pad to multiples of 256)")
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


def _stitch_from_tiles(tiles: list[np.ndarray], meta: dict[str, int]) -> np.ndarray:
    """
    Stitch tile predictions back into a full-resolution background image.

    Parameters
    ----------
    tiles : list[np.ndarray]
        List of predicted (256x256) background tiles.
    meta : dict[str, int]
        Metadata returned by `_split_into_tiles`.

    Returns
    -------
    np.ndarray
        Stitched background prediction, shape (H, W).
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
    model: torch.nn.Module,
    arrays: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    """
    Run the background model on a list of tiles.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded U-Net model on the correct device.
    arrays : list[np.ndarray]
        List of normalized tiles, each (256x256).
    device : torch.device
        Device used for inference.

    Returns
    -------
    list[np.ndarray]
        Model predictions for each tile, each (256x256), dtype float32.
    """
    preds = []
    model.eval()
    logger.debug("Predicting %d tiles on device=%s", len(arrays), device)
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
    line_order: int,
    polyx: int = 1,
    polyy: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the background of an AFM image, return the background & levelled outputs.

    Steps:
    - plane fitting (xy polynomial)
    - tile split → model prediction → stitching
    - denormalisation
    - final line-based correction
    - levelled image = plane-fit image - corrected background

    Parameters
    ----------
    model : torch.nn.Module
        Background U-Net model.
    image : np.ndarray
        Input AFM image, shape (H, W).
    line_order : int
        Polynomial order for the final line fit.
    polyx, polyy : int
        Polynomial orders for the initial plane fit.

    Returns
    -------
    background_linefit : np.ndarray
        Background estimate after line-fit correction.
    levelled : np.ndarray
        Levelled AFM image (image - background).
    """
    # Plane fit and normalise (same as before)
    im_planefit = xyplanefit(image, polyx, polyy)
    imnorm, minval, datarange = normalise(im_planefit)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Processing single image: planefit shape=%s min=%.3f max=%.3f",
            im_planefit.shape,
            im_planefit.min(),
            im_planefit.max(),
        )

    # Split -> predict -> stitch
    arrays, meta = _split_into_tiles(imnorm)
    tiles_pred = _predict_tiles(model, arrays, device=next(model.parameters()).device)
    BGstitched = _stitch_from_tiles(tiles_pred, meta)

    # Denormalise and final line fit
    BG_denorm = denormalise(BGstitched, minval, datarange)
    predictedBG_linefit = linefit(BG_denorm, line_order)

    # Leveled image
    predictedLev = im_planefit - predictedBG_linefit
    return predictedBG_linefit, predictedLev


def subtract_median(image: np.ndarray, *, index: int | None = None) -> np.ndarray:
    """
    Subtract the median value from an image to enforce a zero-height reference.

    This operation removes a constant offset by subtracting the median of all
    pixels in the image. It is intended as a post-processing normalisation step
    (e.g. for AFM image stacks) to ensure consistent relative height references
    between frames, rather than as a physically meaningful background correction.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image array from which the median value will be subtracted.
    index : int or None, optional
        Optional slice or frame index, used only for debug logging. If provided,
        the median value removed from this image is logged at DEBUG level.

    Returns
    -------
    np.ndarray
        Image with its median value subtracted. The returned array has the same
        shape as the input.
    """
    median = np.median(image)
    image = image - median
    if index is not None:
        logger.debug("Slice %d: zeroing median (%.4g)", index, median)
    return image


# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~


def level_ml_bg(
    imarray: np.ndarray,
    model_path: str = "Heath-AFM-Lab/afMLevel-background-unet::background_unet.pth",
    line_order: int = 3,
    background: bool = False,
    zero_median: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """
    Predict and subtract background using the ML U-Net background model.

    This function handles both single AFM frames and multi-frame stacks. Background
    is predicted via tiling, stitching, denormalisation, and an optional final
    line-fit correction. For stacks, each slice is processed independently.

    Parameters
    ----------
    imarray : np.ndarray
        Input AFM image or stack:
        - 2D array: (H, W)
        - 3D array: (N, H, W)
    model_path : str
        Path or HuggingFace identifier for the background model.
        Supports:
        - Local file: "path/model.pth"
        - Repo: "Heath-AFM-Lab/afMLevel-background-unet"
        - Repo::file: "Heath-AFM-Lab/afMLevel-background-unet::background_unet.pth"
        Default downloads the HF model if not cached.
    line_order : int
        Polynomial order used for final line-based correction.
    background : bool
        If True, return the predicted background.
        If False, return the levelled image (image - background).
    zero_median : bool
        If True and `imarray` is a stack, subtract the per-slice median of each
        levelled frame.
    device : str or None
        "cuda", "cpu", or None (auto-select).

    Returns
    -------
    np.ndarray
        If input was 2D: (H, W)
        If input was 3D: (N, H, W)
        dtype float64.
    """
    if imarray.ndim not in (2, 3):
        logger.error("Invalid imarray rank: shape=%s", getattr(imarray, "shape", None))
        raise ValueError(f"imarray must be 2D or 3D, got shape {imarray.shape}")

    # Select device
    torch_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(
        "level_ml_bg start: shape=%s device=%s line_order=%d background=%s",
        imarray.shape,
        torch_device,
        line_order,
        background,
    )

    # Load model once
    n_channels = 1
    model = load_unet_model(model_path, n_channels, UNET_CONFIG, torch_device)

    # Process 2D
    if imarray.ndim == 2:
        logger.debug("Processing single image (H,W)=%s", imarray.shape)
        predictedBG, predictedLev = _process_single_image(model, imarray, line_order)
        if zero_median and not background:
            predictedLev = subtract_median(predictedLev)
        return predictedBG if background else predictedLev

    # Process 3D stack
    N = imarray.shape[0]
    BG_list = []
    Lev_list = []
    for i in range(N):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Processing slice %d/%d", i + 1, N)
        predictedBG, predictedLev = _process_single_image(model, imarray[i], line_order)
        BG_list.append(predictedBG)
        if zero_median:
            predictedLev = subtract_median(predictedLev, index=i)
        Lev_list.append(predictedLev)
    logger.info("All slices processed.")
    BG_stack = np.stack(BG_list, axis=0)
    Lev_stack = np.stack(Lev_list, axis=0)

    logger.info(
        "level_ml_bg done: returning %s stack of shape %s",
        "background" if background else "levelled",
        (BG_stack if background else Lev_stack).shape,
    )
    return BG_stack if background else Lev_stack
