import logging
from typing import List

import cv2
import numpy as np
import torch
from pnanolocz.level_auto import apply_level, apply_level_weighted
from skimage.morphology import (
    dilation,
    disk,
    erosion,
    remove_small_holes,
    remove_small_objects,
)

from afmlevel.unet import load_unet_model
from afmlevel.utils import normalise, remove_small_zeros, swap01, xyplanefit

logger = logging.getLogger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~ MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~

UNET_CONFIG = {
    "filtersize1": 7,
    "filtersize": 7,
    "leakyrelu": False,
    "dropoutprob": 0,
}

# ~~~~~~~~~~~~~~~~~~ HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~


def _predict_mask_256(
    model: torch.nn.Module,
    image_256_norm: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict a (256x256) binary mask from a normalized AFM patch.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded U-Net model already placed on the correct device.
    image_256_norm : np.ndarray
        Normalized 2D AFM image, shape (256, 256), dtype float32 in [0, 1].
    device : torch.device
        Device where inference is executed.
    threshold : float, optional
        Sigmoid cutoff threshold to produce a binary mask.

    Returns
    -------
    np.ndarray
        Binary mask of shape (256, 256), dtype uint8 with values {0, 1}.
    """
    if image_256_norm.shape != (256, 256):
        logger.error(
            "Mask predict got wrong shape: %s (expect 256x256)", image_256_norm.shape
        )
        raise AssertionError(f"Expected (256,256), got {image_256_norm.shape}")

    # Ensure dtype float32 and contiguous layout
    image_256_norm = np.ascontiguousarray(image_256_norm, dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        x = (
            torch.from_numpy(image_256_norm).unsqueeze(0).unsqueeze(0).to(device)
        )  # [1,1,256,256]
        logits = model(x)  # [1,1,256,256]
        probs = torch.sigmoid(logits)  # [1,1,256,256]
        logger.debug("Mask logits->sigmoid done (threshold=%.2f)", threshold)
        binary = (probs > threshold).to(torch.uint8)  # [1,1,256,256] uint8
        mask = (
            binary.squeeze(0).squeeze(0).detach().cpu().numpy()
        )  # (256,256), values {0,1}
    return mask


def _process_single_image_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    polyx: int = 1,
    polyy: int = 1,
    out_min_size: int = 30,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Process a single AFM image using the U-Net mask model.

    Steps:
    - plane fit (xypolynomial)
    - resize to 256x256
    - normalize
    - predict mask using U-Net
    - resize back to original resolution
    - optional small-object removal
    - enforce uint8 {0,1}

    Parameters
    ----------
    image : np.ndarray
        Input AFM heightmap, shape (H, W).
    model : torch.nn.Module
        Loaded U-Net mask model.
    polyx, polyy : int
        Polynomial orders for plane fitting.
    out_min_size : int
        Minimum connected size of zeros to keep.
    threshold : float
        Probability threshold for binarization.

    Returns
    -------
    np.ndarray
        Binary mask (H, W) with values {0,1}, dtype uint8.
    """
    H, W = image.shape
    # plane fit
    im_planefit = xyplanefit(image, polyx, polyy)
    # resize to 256x256
    im_resized = cv2.resize(im_planefit, (256, 256), interpolation=cv2.INTER_NEAREST)
    # normalize
    im_norm, _, _ = normalise(im_resized)

    # predict on 256x256 normalized
    device = next(model.parameters()).device
    mask_256 = _predict_mask_256(model, im_norm, device=device, threshold=threshold)

    # resize mask back to original (nearest-neighbor keeps labels intact)
    mask_back = cv2.resize(mask_256, (W, H), interpolation=cv2.INTER_NEAREST).astype(
        np.uint8
    )

    # post-processing
    mask_swapped = swap01(mask_back)  # keeps {0,1}, returns np.uint8 or bool
    mask_final = remove_small_zeros(
        mask_swapped, min_size=out_min_size
    )  # returns {0,1}

    # Ensure uint8 {0,1}
    return (mask_final > 0).astype(np.uint8)


# ~~~~~~~~~~~~~~~~~~ ML MASK GENERATION ~~~~~~~~~~~~~~~~~~~~~


def ml_mask(
    imarray: np.ndarray,
    model_path: str = "Heath-AFM-Lab/afMLevel-mask-unet::mask_unet.pth",
    device: str | None = None,
    threshold: float = 0.5,
    polyx: int = 1,
    polyy: int = 1,
    min_size: int = 30,
) -> np.ndarray:
    """
    Compute a background mask (1 = background, 0 = feature) using the ML U-Net model.

    Parameters
    ----------
    imarray : np.ndarray
        Input AFM image or stack:
        - 2D array: (H, W)
        - 3D array: (N, H, W)
    model_path : str
        Path or HuggingFace model specifier. Supports:
        - local file: "/path/model.pth"
        - HuggingFace repo: "user/repo"
        - repo + file: "user/repo::filename.pth"
    device : str or None
        "cpu" or "cuda". If None, chooses automatically.
    threshold : float
        Sigmoid threshold used during mask prediction.
    polyx, polyy : int
        Plane-fit polynomial orders.
    min_size : int
        Minimum zero-region size to preserve.

    Returns
    -------
    np.ndarray
        If input is 2D: returns (H, W) mask.
        If input is 3D: returns (N, H, W) mask stack.
    """
    if imarray.ndim not in (2, 3):
        logger.error(
            "Invalid imarray rank for ml_mask: shape=%s",
            getattr(imarray, "shape", None),
        )
        raise ValueError(f"imarray must be 2D or 3D, got {imarray.shape}")

    # Ensure float32 input for consistent numeric path
    if imarray.dtype != np.float32:
        imarray = imarray.astype(np.float32, copy=False)

    torch_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info(
        "ml_mask start: shape=%s device=%s threshold=%.2f min_size=%d",
        imarray.shape,
        torch_device,
        threshold,
        min_size,
    )

    # Load model once
    n_channels = 1
    model = load_unet_model(model_path, n_channels, UNET_CONFIG, torch_device)

    if imarray.ndim == 2:
        return _process_single_image_mask(
            model,
            imarray,
            polyx=polyx,
            polyy=polyy,
            out_min_size=min_size,
            threshold=threshold,
        )

    # 3D stack
    N = imarray.shape[0]
    out_masks: List[np.ndarray] = []
    for i in range(N):
        mask_i = _process_single_image_mask(
            model,
            imarray[i],
            polyx=polyx,
            polyy=polyy,
            out_min_size=min_size,
            threshold=threshold,
        )
        out_masks.append(mask_i)

    return np.stack(out_masks, axis=0)


def _perimeter_remove(bw2d: np.ndarray) -> np.ndarray:
    """
    Return the perimeter pixels of a binary image.

    Direct analogue of MATLAB `bwmorph(bw, 'remove')`:
    removes interior pixels and leaves only boundary pixels.

    Parameters
    ----------
    bw2d : np.ndarray
         2D binary array, dtype can be bool or uint8 {0,1} or bool.

    Returns
    -------
    np.ndarray
        2D binary array of the same shape, dtype bool (perimeter mask)
    """
    # Ensure boolean for morphology
    bw_bool = bw2d > 0

    # 8-connected 3×3 footprint
    footprint = np.ones((3, 3), dtype=bool)

    # Erode the foreground
    eroded = erosion(bw_bool, footprint=footprint)

    # Perimeter = fg AND NOT(eroded)
    perimeter_bool = bw_bool & (~eroded)

    # Return as bool
    return perimeter_bool


def ml_edges(
    imarray: np.ndarray,
    model_path: str = "Heath-AFM-Lab/afMLevel-mask-unet::mask_unet.pth",
    device: str | None = None,
    threshold: float = 0.5,
    polyx: int = 1,
    polyy: int = 1,
    min_size: int = 30,
    area_thresh_objects: int = 100,
    area_thresh_holes: int = 50,
    dilate_disk_radius: int = 3,
) -> np.ndarray:
    """
    Generate an edge mask from the ML U-Net mask.

    Applies morphology steps analogous to MATLAB's `bwmorph` and `bwareaopen`.

    Parameters
    ----------
    imarray : np.ndarray
        Input AFM image (H,W) or stack (N,H,W).
    model_path : str
        HuggingFace or local U-Net model path.
    device : str or None
        Execution device passed to `ml_mask`.
    threshold : float
        Sigmoid threshold for mask inference.
    polyx, polyy : int
        Polynomial orders for plane fitting (inside ml_mask).
    min_size : int
        Minimum object size for postprocessing.
    area_thresh_objects : int
        Minimum object size to keep during morphological cleanup.
    area_thresh_holes : int
        Maximum hole size to fill.
    dilate_disk_radius : int
        Disk radius for dilation.

    Returns
    -------
    np.ndarray
        Binary uint8 mask (same shape as input), where:
        - 1 = background/interior
        - 0 = detected edges.
    """
    # 1)  Normalize to NHW
    imarray = np.array(imarray, np.float32, copy=False)
    input_was_2d = imarray.ndim == 2
    if input_was_2d:
        img_nhw = imarray[None, ...]  # (1, H, W)
    elif imarray.ndim == 3:
        img_nhw = imarray  # assume NHW
    else:
        raise ValueError("imarray must be 2D (H,W) or 3D (N,H,W)")

    # 2)  Generate ML mask with the `ml_mask` function.
    ml_mask_result = ml_mask(
        img_nhw, model_path, device, threshold, polyx, polyy, min_size
    )

    # 3) Invert before processing
    ml_mask_invert = 1 - ml_mask_result  # NHW, int {0,1}

    # 4) Apply MATLAB operations per-slice
    N, H, W = ml_mask_invert.shape
    out = np.zeros((N, H, W), dtype=bool)  # boolean for morphology

    # correct new skimage semantics:
    max_obj = max(0, area_thresh_objects - 1)
    max_hole = max(0, area_thresh_holes - 1)

    for i in range(N):
        # Convert to boolean **here** (only inside morphology block)
        bool_mask = ml_mask_invert[i].astype(bool)

        # bwmorph('remove')
        bool_mask = _perimeter_remove(bool_mask)

        # bwareaopen(BW, 100) — remove small objects
        if max_obj > 0:
            bool_mask = remove_small_objects(
                bool_mask, max_size=max_obj, connectivity=2
            )

        # ~bwareaopen(~BW, 50) — fill small holes
        if max_hole > 0:
            bool_mask = remove_small_holes(bool_mask, max_size=max_hole, connectivity=2)

        # imdilate(BW, strel('disk',3))
        bool_mask = dilation(bool_mask, footprint=disk(dilate_disk_radius))

        # second round: bwareaopen + ~bwareaopen(~·)
        if max_obj > 0:
            bool_mask = remove_small_objects(
                bool_mask, max_size=max_obj, connectivity=2
            )
        if max_hole > 0:
            bool_mask = remove_small_holes(bool_mask, max_size=max_hole, connectivity=2)

        # Store boolean
        out[i] = bool_mask

    # 5) Final inversion to match MATLAB's ~imgt output
    result_bool = ~out

    # Return mask as uint8 {0,1}
    result = result_bool.astype(np.uint8)
    # Restore original rank
    if input_was_2d:
        return result[0]
    return result


# ~~~~~~~~~~~~~~~~~~ ML LEVELLING ROUTINES ~~~~~~~~~~~~~~~~~~~~~

# Create a routines for iteratively applying the model and plane fits.
# Uses the format of the `ROUTINES` dictionary in `pnanolocz.level_auto`.
DEFAULT_ML_ROUTINES = {
    "iterative ML mask": [
        # Initial plane fit to raw image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        # First model application to get mask
        {
            "func": ml_mask,
            "invert": True,
        },
        # Second plane fit to masked image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        # Second model application to get mask
        {
            "func": ml_mask,
            "invert": True,
        },
        # Third plane fit to masked image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        # Third model application to get mask
        {
            "func": ml_mask,
            "invert": True,
        },
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        # Plane fit in x direction to masked image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 0,
            "method": "plane",
        },
        # Fourth model application to get mask
        {
            "func": ml_mask,
            "invert": True,
        },
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        # Second order plane fit in x direction to masked image
        {
            "func": apply_level,
            "polyx": 2,
            "polyy": 0,
            "method": "plane",
        },
    ],
    "ML mask": [
        # Initial plane fit to raw image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        # First model application to get mask
        {
            "func": ml_mask,
            "invert": True,
        },
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        # Second plane fit to masked image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
    ],
    "multi-plane-ML-it": [
        # Initial plane fit to raw image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        # First model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Second model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Another second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Third model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Another second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        # Fourth model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Another second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
    ],
    "multi-plane-ML": [
        # Initial plane fit to raw image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        # First model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
    ],
    "multi-plane-ML-it-line": [
        # Initial plane fit to raw image
        {
            "func": apply_level,
            "polyx": 1,
            "polyy": 1,
            "method": "plane",
        },
        # Median line fit
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        # First model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Second model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Another second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Third model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Another second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
        # Fourth model application to get mask
        {
            "func": ml_edges,
            "invert": True,
        },
        # Another second order weighted plane fit to masked image
        {"func": apply_level_weighted, "polyx": 2, "polyy": 2, "method": "plane"},
        # Median line fit to masked image
        {
            "func": apply_level,
            "polyx": 0,
            "polyy": 0,
            "method": "med_line",
        },
    ],
}

# ~~~~~~~~~~~~~~~~~~~~~~~ LEVELING FUNCTION ~~~~~~~~~~~~~~~~~~~~~~~~~~


def level_ml_mask(
    imarray: np.ndarray,
    model_path: str = "Heath-AFM-Lab/afMLevel-mask-unet::mask_unet.pth",
    device: str | None = None,
    threshold: float = 0.5,
    min_size: int = 30,
    method: str = "iterative ML mask",
    ml_routines: dict | None = None,
) -> np.ndarray:
    """
    AFM leveling using a learned mask and routines of plane/median fits.

    The routines are defined in the `DEFAULT_ML_ROUTINES` dictionary, which specifies
    sequences of operations (plane fits, median line fits, and model applications) to
    be applied iteratively. The model applications use the `ml_mask` and `ml_edges`
    functions defined above, which apply the trained U-Net model to predict masks and
    edge masks, respectively. The leveling functions are imported from
    `pnanolocz.level_auto` (https://github.com/derollins/Python-Nanolocz-Library)
    and are applied according to the specified routine.

    Available methods (keys of `DEFAULT_ML_ROUTINES`):
    - "iterative ML mask": multiple iterations of ml_mask and plane fits, ending with
      a median line fit.
    - "ML mask": single ml_mask application followed by a median line fit and plane
      fit.
    - "multi-plane-ML-it": multiple iterations of ml_edges and second-order weighted
      plane fits, ending with a median line fit.
    - "multi-plane-ML": single ml_edges application followed by a second-order weighted
      plane fit and a median line fit.
    - "multi-plane-ML-it-line": similar to "multi-plane-ML-it" but with additional
      median line fit after th initial plane fit.

    These routines are adapted from the auto level routines from the Nanolocz software
    libraries (Python-Nanolocz-Library version:
    https://github.com/derollins/Python-Nanolocz-Library). The "iterative ML mask"
    routine is a direct analogue of the original iterative routines but with the ML
    generated mask replacing the histogram based masks in the original. The multi-plane
    routines are adaptations of the mulit-plane routines an use edge masks created from
    the ML model generated mask, processed to mask region edges.

    **Mask polarity**: `ml_mask` return a binary *background* mask
    (1 = background, 0 = feature). Downstream filters (e.g., `apply_level`) expect a
    boolean *foreground* mask (True = foreground). Therefore, the routine **inverts**
    the model mask before passing it to filters so that True denotes foreground.

    Parameters
    ----------
    imarray : np.ndarray
        Input image or stack.
        - 2D: shape (H, W), processed as a single-frame stack of shape (1, H, W)
        - 3D: shape (N, H, W), processed frame-by-frame
        Internally cast to float64 for leveling, and to float32 for model inference.
    model_path : str
        Path or HuggingFace identifier for the U-Net mask model.
        Default is:
        "Heath-AFM-Lab/afMLevel-mask-unet::mask_unet.pth"
        which downloads the model from HuggingFace if not cached.
    device : Optional[str]
        'cuda' or 'cpu'. If None, auto-selects CUDA if available.
    threshold : float
        Threshold used inside `ml_mask` for binarisation after sigmoid.
    min_size : int
        Minimum component size used by `ml_mask` in post-processing (e.g.,
        remove_small_zeros).

    Returns
    -------
    np.ndarray
        Processed image or stack:
        - shape (H, W) if input was 2D
        - shape (N, H, W) if input was 3D
        dtype float64 (matching typical leveling pipeline expectations)

    Notes
    -----
    - The raw outputs of `ml_mask` and `ml_edges` (binary) is converted to a boolean
      mask then **inverted** so that True = foreground (matching the mask polarity
      expected by pnanolocz filters).
    - If the input image was 2D, the internal batch dimension is removed before
      returning.
    """
    routines = DEFAULT_ML_ROUTINES if ml_routines is None else ml_routines

    logger.info(
        "level_ml_mask start: method=%s device=%s threshold=%.2f min_size=%d",
        method,
        device,
        threshold,
        min_size,
    )

    if method not in routines:
        logger.error("Unknown routine method: %s", method)
        raise ValueError(
            "Unknown routine method. Available methods: " + ", ".join(routines.keys())
        )

    # ----- Normalise input shape and dtype -----
    arr = np.asarray(imarray)
    # Cast uint8 -> float64 in [0,1]; otherwise float64
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = arr.astype(np.float64, copy=False)

    # Ensure (N, H, W)
    was_2d = False
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
        was_2d = True
    elif arr.ndim == 3:
        # Handle (H, W, 1) or (H, W, C) gracefully as single-frame
        if arr.shape[-1] == 1:  # (H, W, 1)
            arr = arr[..., 0][np.newaxis, ...]
            was_2d = True
        elif arr.shape[0] == 1 and arr.shape[-1] != 1:  # (1, H, W, C?) oddball
            arr = arr[0]
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
                was_2d = True
            else:
                raise ValueError("Unexpected 3D/4D shape for imarray")
        # else: assume proper (N, H, W)
    else:
        raise ValueError("imarray must be 2D or 3D")

    # Device selection (pass through to ml_mask)
    torch_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    result = arr.copy()

    steps = routines[method]
    N = result.shape[0]

    for i in range(N):
        img = result[i]
        fg_mask_bool = None  # foreground mask, boolean

        for step in steps:
            func = step["func"]

            if func is ml_mask:
                logger.debug(
                    "Routine step: ml_mask (invert=%s)", step.get("invert", True)
                )
                # --- Call ml_mask on the current image ---
                # ml_mask handles float conversion & normalisation internally;
                #  we pass float32 for speed
                img_for_model = np.asarray(img, dtype=np.float32, order="C")
                # Expect ml_mask to return (H,W) uint8 {0,1} or bool mask
                bg_mask = ml_mask(
                    img_for_model,
                    model_path=model_path,
                    device=str(torch_device),
                    threshold=threshold,
                    min_size=min_size,
                )
                bg_mask = np.asarray(bg_mask)

                # Convert to boolean
                if bg_mask.dtype == np.uint8 or np.issubdtype(
                    bg_mask.dtype, np.integer
                ):
                    bg_mask = bg_mask.astype(bool)
                elif np.issubdtype(bg_mask.dtype, np.floating):
                    bg_mask = bg_mask > 0.5
                elif bg_mask.dtype != bool:
                    bg_mask = bg_mask.astype(bool)

                # Invert to get foreground (True = features)
                if step.get("invert", True):
                    fg_mask_bool = ~bg_mask
                else:
                    fg_mask_bool = bg_mask
                continue
            elif func is ml_edges:
                logger.debug(
                    "Routine step: ml_edges (invert=%s)", step.get("invert", True)
                )
                # --- Call ml_edges on the current image ---
                # ml_edges handles float conversion & normalisation internally;
                #  we pass float32 for speed
                img_for_model = np.asarray(img, dtype=np.float32, order="C")
                # Expect ml_edges to return (H,W) uint8 {0,1} or bool mask
                edge_mask = ml_edges(
                    img_for_model,
                    model_path=model_path,
                    device=str(torch_device),
                    threshold=threshold,
                    min_size=min_size,
                )
                edge_mask = np.asarray(edge_mask)

                # Convert to boolean
                if edge_mask.dtype == np.uint8 or np.issubdtype(
                    edge_mask.dtype, np.integer
                ):
                    edge_mask = edge_mask.astype(bool)
                elif np.issubdtype(edge_mask.dtype, np.floating):
                    edge_mask = edge_mask > 0.5
                elif edge_mask.dtype != bool:
                    edge_mask = edge_mask.astype(bool)

                # Invert to get foreground (True = features)
                if step.get("invert", True):
                    fg_mask_bool = ~edge_mask
                else:
                    fg_mask_bool = edge_mask
                continue

            # --- Otherwise it's a leveling step ---
            # Pass the foreground mask to apply_level
            # NOTE: apply_level is assumed to accept a mask where True means foreground

            logger.debug(
                "Routine step: %s with args=%s",
                getattr(func, "__name__", str(func)),
                {k: v for k, v in step.items() if k not in ("func", "invert")},
            )
            img = func(
                img,
                mask=fg_mask_bool,
                **{k: v for k, v in step.items() if k not in ("func", "invert")},
            )

        # Save back the processed frame
        result[i] = img

    # Return original dimensionality
    logger.info(
        "level_ml_mask done: returning %s",
        "2D frame" if was_2d else f"stack with shape {result.shape}",
    )
    return result[0] if was_2d else result
