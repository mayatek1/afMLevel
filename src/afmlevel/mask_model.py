from typing import Any, List, Optional

import cv2
import numpy as np
import torch
from pnanolocz_lib.level_auto import apply_level

from afmlevel.unet import load_unet_model
from afmlevel.utils import normalise, remove_small_zeros, swap01, xyplanefit

# from afmlevel.post import swap01, remove_small_zeros  # adjust your import path
# Ensure swap01 and remove_small_zeros are imported from wherever they live

# ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~

UNET_CONFIG = {
    "filtersize1": 7,
    "filtersize": 7,
    "leakyrelu": False,
    "dropoutprob": 0,
}


def _predict_mask_256(
    model: torch.nn.Module,
    image_256_norm: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict a binary mask from a single normalized 256x256 image (float32 in [0,1]).

    Returns (256, 256) np.uint8 array with {0,1}.
    """
    assert image_256_norm.shape == (
        256,
        256,
    ), f"Expected (256,256), got {image_256_norm.shape}"
    # Ensure dtype float32 and contiguous layout
    image_256_norm = np.ascontiguousarray(image_256_norm, dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        x = (
            torch.from_numpy(image_256_norm).unsqueeze(0).unsqueeze(0).to(device)
        )  # [1,1,256,256]
        logits = model(x)  # [1,1,256,256]
        probs = torch.sigmoid(logits)  # [1,1,256,256]
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
    Process one 2D image.

    plane-fit -> resize(256x256) -> normalise -> predict
    -> resize back -> swap01 -> remove_small_zeros

    Returns a binary mask with the same HxW shape as the input (dtype uint8,
    values {0,1}).
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


def ml_mask(
    imarray: np.ndarray,
    model_path: str,
    device: Optional[str] = None,
    threshold: float = 0.5,
    polyx: int = 1,
    polyy: int = 1,
    min_size: int = 30,
) -> np.ndarray:
    """
    Apply the trained U-Net mask model to a single AFM image (H,W) or a stack (N,H,W).

    For a 2D input, returns a (H,W) binary mask.
    For a 3D input, returns a (N,H,W) stack of binary masks.

    Parameters
    ----------
    imarray : np.ndarray
        2D image (H,W) or 3D stack (N,H,W).
    model_path : str
        Path to .pth (state_dict) file.
    device : Optional[str]
        'cuda' or 'cpu'; if None, auto-select CUDA if available.
    threshold : float
        Sigmoid probability threshold for binarisation.
    polyx, polyy : int
        Plane-fit polynomial orders.
    min_size : int
        Minimum 'hole' size to fill in remove_small_zeros (post-processing).

    Returns
    -------
    np.ndarray
        Same leading shape as input. Binary masks with values {0,1}, dtype uint8.
    """
    if imarray.ndim not in (2, 3):
        raise ValueError(f"imarray must be 2D or 3D, got {imarray.shape}")

    # Ensure float32 input for consistent numeric path
    if imarray.dtype != np.float32:
        imarray = imarray.astype(np.float32, copy=False)

    torch_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
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


def level_ml_mask(
    imarray: np.ndarray,
    model_path: str,
    device: Optional[str] = None,
    threshold: float = 0.5,
    min_size: int = 30,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    AFM leveling using a learned mask and a routine of plane/median fits.

    This routine alternates between predicting a mask with a trained model
    (`ml_mask`) and applying leveling operations (`apply_level`) while
    respecting the mask. This is based on the iterative 1nm routine from the
    Nanolocz software library (Python-Nanolocz-Library version:
    https://github.com/derollins/Python-Nanolocz-Library).

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
        Path to the trained model used by `ml_mask`.
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
    - The raw output of `ml_mask` (probabilities or boolean) is converted to
      a boolean mask by thresholding at 0.5 if floating, then **inverted** so that
      True = foreground (matching the mask polarity expected by pnanolocz_lib filters).
    - If `ml_mask` returns shape (1, H, W), it is squeezed to (H, W).
    """
    # Create a routine for iteratively applying the model and plane fits.
    # Uses the format of the `ROUTINES` dictionary in `pnanolocz_lib.level_auto`.
    ml_routine = {
        "iterative ML mask": [
            # First model application to get mask
            {
                "func": ml_mask,
                "model_path": model_path,
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
                "model_path": model_path,
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
                "model_path": model_path,
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
                "model_path": model_path,
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
    }

    if "iterative ML mask" not in ml_routine:
        raise ValueError("Unknown routine 'iterative ML mask'")

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
    steps = ml_routine["iterative ML mask"]
    N = result.shape[0]

    for i in range(N):
        img = result[i]
        fg_mask_bool = None  # foreground mask, boolean

        for step in steps:
            func = step["func"]

            if func is ml_mask:
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

            # --- Otherwise it's a leveling step ---
            # Pass the foreground mask to apply_level
            # NOTE: apply_level is assumed to accept a mask where True means foreground
            img = func(
                img,
                mask=fg_mask_bool,
                **{k: v for k, v in step.items() if k not in ("func", "invert")},
            )

        # Save back the processed frame
        result[i] = img

    # Return original dimensionality
    return result[0] if was_2d else result
