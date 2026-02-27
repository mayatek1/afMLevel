import os
from typing import Any

import cv2
import numpy as np
import torch

from afmlevel.unet import UNet
from afmlevel.utils import normalise, remove_small_zeros, swap01, xyplanefit
from pnanolocz_lib.level_auto import apply_level

# ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~

UNET_CONFIG = dict(
    filtersize1=7,
    filtersize=7,
    leakyrelu=False,
    dropoutprob=0,
)


def level_ml_mask_auto(
    imarray: np.ndarray, model_path: str
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    AFM leveling using a learned mask and a routine of plane/median fits.

    This routine alternates between predicting a mask with a trained model
    (`applymodel_mask`) and applying leveling operations (`apply_level`) while
    respecting the mask. This is based on the iterative 1nm routine from the
    Nanolocz software library (Python-Nanolocz-Library version:
    https://github.com/derollins/Python-Nanolocz-Library).

    **Mask polarity**: `applymodel_mask` return a binary *background* mask
    (1 = background, 0 = feature). Downstream filters (e.g., `apply_level`) expect a
    boolean *foreground* mask (True = foreground). Therefore, the routine **inverts**
    the model mask before passing it to filters so that True denotes forground.

    Parameters
    ----------
    imarray : np.ndarray
        Input image or stack.
        - 2D: shape (H, W), processed as a single-frame stack of shape (1, H, W)
        - 3D: shape (N, H, W), processed frame-by-frame
        Internally cast to float64 for leveling, and to float32 for model inference.
    model_path : str
        Path to the trained model used by `applymodel_mask`.

    Returns
    -------
    np.ndarray
        Processed image or stack:
        - shape (H, W) if input was 2D
        - shape (N, H, W) if input was 3D

    Notes
    -----
    - The raw output of `applymodel_mask` (probabilities or boolean) is converted to
      a boolean mask by thresholding at 0.5 if floating, then **inverted** so that
      True = forground (matching the mask polarity expected by filters).
    - If `applymodel_mask` returns shape (1, H, W), it is squeezed to (H, W).
    """
    # Create a routine for iteratively applying the model and plane fits.
    # Uses the format of the `ROUTINES` dictinary in `pnanolocz_lib.level_auto`.
    ml_routine = {
        "iterative ML mask": [
            # First model application to get mask
            {
                "func": applymodel_mask,
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
                "func": applymodel_mask,
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
                "func": applymodel_mask,
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
                "func": applymodel_mask,
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
    imarray = np.asarray(imarray)

    # Cast uint8 to float64 in range [0, 1]
    if imarray.dtype == np.uint8:
        imarray = imarray.astype(np.float64) / 255.0
    else:
        imarray = imarray.astype(np.float64)

    # Normalise dimensions- Ensure imarray is 3D (N, H, W) for consistent processing
    if imarray.ndim == 2:
        imarray = imarray[np.newaxis, :, :]
        was_2d = True

    elif imarray.ndim == 3:
        # Determine whether this is (N, H, W) or (H, W, C)
        # AFM data is never channel-last, so treat as stack only if first dimension is small
        if imarray.shape[0] == 1 and imarray.shape[-1] != 1:
            # Probably a color or weird channel image → collapse last axis
            imarray = imarray[..., 0]
            imarray = imarray[np.newaxis, :, :]
            was_2d = True
        elif imarray.shape[-1] == 1:
            # Interpret (H, W, 1) as a single frame
            imarray = imarray[..., 0]  # remove channel
            imarray = imarray[np.newaxis, :, :]
            was_2d = True
        else:
            # This is a real stack (N, H, W)
            was_2d = False

    else:
        raise ValueError("imarray must be 2D or 3D")

    result = imarray.copy()
    steps = ml_routine["iterative ML mask"]
    frames = range(imarray.shape[0])

    for i in frames:
        img = result[i].copy()
        mask = None
        for _idx, step in enumerate(steps):
            func = step["func"]
            params = {k: v for k, v in step.items() if k != "func"}

            if func is applymodel_mask:
                model_path = params["model_path"]

                # 1) Ensure consistent dtype/range
                img_for_model = np.asarray(img, dtype=np.float32, order="C")
                mask_raw = applymodel_mask(img_for_model, model_path)

                # 3) Convert to NumPy 2D
                mask = np.asarray(mask_raw)
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]

                # 4) Convert floats → boolean safely
                if mask.dtype != bool:
                    if np.issubdtype(mask.dtype, np.floating):
                        mask = mask > 0.5
                    else:
                        mask = mask.astype(bool)

                # 5) Only invert if requested
                if params.get("invert", True):
                    mask = ~mask

                continue  # go to next step
            # Generic path for all other steps (unchanged)
            img = func(
                img,
                mask=mask,
                **{k: v for k, v in params.items() if k not in ("args", "invert")},
            )

            result[i] = img.copy()

    return np.asarray(result[0]) if was_2d else np.asarray(result)


def applymodel_mask(imarray, model_path):

    dim = (256, 256)  # dimensions for image resize
    polyx = 1
    polyy = 1  # order of polynomial plane fit to initially apply to raw image
    n_channels = 1
    # won't be changing which model is used once we have found the best one, so this is defined in the function

    # ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # preprocess the image
    imarray = xyplanefit(imarray, polyx, polyy)  # apply plane fit
    imarray_r = cv2.resize(imarray, dim, interpolation=cv2.INTER_NEAREST)
    imnorm, minval, maxval = normalise(imarray_r)  # normalise to values between 0 and 1

    def load_model(model_path: str, n_channels: int, device: str):
        model = UNet(n_channels, **UNET_CONFIG)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    loaded_model = load_model(model_path, n_channels, device)
    # print("Model loaded successfully.")

    def predict_on_image(model, image, device):
        model.eval()
        with torch.no_grad():
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            output = model(image_tensor)
            output_s = torch.sigmoid(output)
            output_b = (output_s > 0.5).float()

        return output_b.squeeze(0).cpu()

    # apply model to image
    predictedmask = predict_on_image(
        loaded_model, imnorm, device
    )  # shape [1, 256, 256]
    predictedmask = predictedmask.squeeze(0)  # shape [256, 256]
    predictedmask = predictedmask.numpy()
    predictedmask_resize = cv2.resize(
        predictedmask,
        (imarray.shape[1], imarray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    predictedmask_swap = swap01(predictedmask_resize)
    predictedmask_final = remove_small_zeros(predictedmask_swap, min_size=30)

    return predictedmask_final


def applymodel_mask_stack(imarray, model_path):

    dim = (256, 256)  # dimensions for image resize
    if imarray.ndim == 3:
        originaldim = (
            imarray.shape[2],
            imarray.shape[1],
        )  # have to use in this order for resize
    else:
        originaldim = (imarray.shape[1], imarray.shape[0])
    polyx = 1
    polyy = 1  # order of polynomial plane fit to initially apply to raw image
    n_channels = 1
    # won't be changing which model is used once we have found the best one, so this is defined in the function

    if imarray.ndim == 3:
        lenstack = imarray.shape[0]
    else:
        lenstack = 1

    minvallist = []
    datarangelist = []
    imnormlist = []
    for i in range(lenstack):
        if imarray.ndim == 3:
            imarrayi = imarray[i, :, :]
        else:
            imarrayi = imarray
        imarray_planefit = xyplanefit(imarrayi, polyx, polyy)  # apply plane fit
        imarray_r = cv2.resize(imarray_planefit, dim, interpolation=cv2.INTER_NEAREST)
        imnorm, minval, datarange = normalise(imarray_r)

        imnormlist.append(imnorm)
        minvallist.append(minval)
        datarangelist.append(datarange)

    def load_model(model_path: str, n_channels: int, device: str):
        model = UNet(n_channels, **UNET_CONFIG)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    loaded_model = load_model(model_path, n_channels, device)
    # print("Model loaded successfully.")

    def predict_on_image(model, image, device):
        model.eval()
        with torch.no_grad():
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            output = model(image_tensor)
            output_s = torch.sigmoid(output)
            output_b = (output_s > 0.5).float()

        return output_b.squeeze(0).cpu()

    # apply model to image
    predictedmasklist = []
    for imnorm, minval, datarange in zip(
        imnormlist, minvallist, datarangelist, strict=False
    ):
        maskarray = predict_on_image(
            loaded_model, imnorm, device
        )  # shape [1, 256, 256]
        maskarray_2D = maskarray.squeeze(0)  # shape [256, 256]
        maskarray_np = maskarray_2D.numpy()

        maskarray_resize = cv2.resize(
            maskarray_np, originaldim, interpolation=cv2.INTER_NEAREST
        )
        maskarray_swap = swap01(maskarray_resize)
        mask_final = remove_small_zeros(maskarray_swap, min_size=30)
        predictedmasklist.append(mask_final)

    return predictedmasklist
