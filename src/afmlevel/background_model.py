import math
import os
from typing import List

import numpy as np
import torch

from afmlevel.unet import UNet
from afmlevel.utils import denormalise, linefit, normalise, xyplanefit

# ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~

UNET_CONFIG = dict(
    filtersize1=9,
    filtersize=9,
    leakyrelu=False,
    dropoutprob=0,
)


def applymodel_bg(
    imarray: np.ndarray,
    lineorder: int,
    model_path: str,
    background: bool = False,
) -> np.ndarray:
    """
    Apply the trained U-Net model to predict background and level an AFM image.

    Parameters
    ----------
    imarray : np.ndarray
        Input raw AFM image to level.
    lineorder : int
        Polynomial order for the final line fit.
    model_path : str
        String of the path to the trained model (.pth file).

    Returns
    -------
    (np.ndarray)
        The final levelled AFM image or the predicted background after the final linefit if background=True.
    """
    # TODO: remove the preprocessing step from the function that applies the model
    # or parametise.
    polyx = 1
    polyy = 1  # order of polynomial plane fit to initially apply to raw image
    n_channels = 1

    # preprocess the image
    imarray = xyplanefit(imarray, polyx, polyy)  # apply plane fit
    imnorm, minval, datarange = normalise(imarray)

    h, w = imarray.shape

    # create mirrored array
    mirror_lr = np.concatenate((imnorm, np.fliplr(imnorm)), axis=1)
    mirror_all = np.concatenate((mirror_lr, np.flipud(mirror_lr)), axis=0)

    if imarray.shape == (256, 256):
        arrays = [imnorm]

    elif imarray.shape == (512, 512):
        arrays = [
            imnorm[0::2, 0::2],
            imnorm[0::2, 1::2],
            imnorm[1::2, 0::2],
            imnorm[1::2, 1::2],
        ]

    else:  # elif h % 256  != 0 or w % 256 != 0:

        # create downsampled 256x256 arrays
        hr = 256 * math.ceil(h / 256)  # height rounded up to nearest multiple of 256
        wr = 256 * math.ceil(w / 256)  # width rounded up to nearest multiple of 256
        hj = int(hr / 256)  # height pixel jump
        wj = int(wr / 256)  # width pixel jump

        arrays = []
        for i in range(hj):
            for j in range(wj):
                array = mirror_all[i:hr:hj, j:wr:wj]
                array = array[:256, :256]
                arrays.append(array)

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

    def predict_on_image(model, arrays, device):
        model.eval()
        BGwindows = []
        with torch.no_grad():
            for image in arrays:
                image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                BGwindow = model(image_tensor)
                BGwindows.append(BGwindow.squeeze(0).cpu())
        return BGwindows

    # apply model to image
    BGarrays = predict_on_image(loaded_model, arrays, device)  # shape [1, 256, 256]
    BGarrays_2D = [array.squeeze(0) for array in BGarrays]  # shape [256, 256]
    BGarrays_np = [array.numpy() for array in BGarrays_2D]

    if imarray.shape == (256, 256):
        BGstitched = BGarrays_np[0]
    elif imarray.shape == (512, 512):
        BGstitched = np.zeros((h, w), dtype=BGarrays_np[0].dtype)
        # Place the BG pixels back in their original positions
        BGstitched[0::2, 0::2] = BGarrays_np[0]  # Top-left positions
        BGstitched[0::2, 1::2] = BGarrays_np[1]  # Top-right positions
        BGstitched[1::2, 0::2] = BGarrays_np[2]  # Bottom-left positions
        BGstitched[1::2, 1::2] = BGarrays_np[3]  # Bottom-right positions
    else:
        BGstitched = np.zeros_like(mirror_all[:hr, :wr])
        for idx, array in enumerate(BGarrays_np):
            i = idx // wj  # row index in the sampling grid
            j = idx % wj  # column index in the sampling grid
            BGstitched[i:hr:hj, j:wr:wj] = array

        # Crop back to original dimensions
        BGstitched = BGstitched[:h, :w]

    BGwindows_denorm = denormalise(BGstitched, minval, datarange)

    predictedBG_linefit = linefit(BGwindows_denorm, lineorder)

    predictedLev = imarray - predictedBG_linefit

    if background is True:
        return predictedBG_linefit
    else:
        return predictedLev


def applymodel_bg_stack(
    imarray,
    lineorder,
    model_path,
    background: bool = False,
) -> List[np.ndarray]:
    """
    Apply the trained U-Net model to predict background and level an AFM image stack.

    Parameters
    ----------
    imarray : np.ndarray
        Input raw AFM image stack to level.
    lineorder : int
        Polynomial order for the final line fit.
    model_path : str
        String of the path to the trained model (.pth file).
    background : bool, optional
        If True, return the predicted background after the final line fit instead of the levelled image

    Returns
    -------
    List [np.ndarray]
        The the final levelled AFM images in a list or the predicted backgrounds in a list after the final
        linefit if background=True.
    """
    if imarray.ndim == 3:
        originaldim = imarray.shape[1:]
    else:
        originaldim = imarray.shape
    polyx = 1
    polyy = 1  # order of polynomial plane fit to initially apply to raw image
    n_channels = 1
    lineorder = lineorder

    # preprocess each image
    if imarray.ndim == 3:
        lenstack = imarray.shape[0]
    else:
        lenstack = 1

    arrayslist = []
    minvallist = []
    datarangelist = []
    imarray_planefit_list = []

    for i in range(lenstack):
        if imarray.ndim == 3:
            imarrayi = imarray[i, :, :]
        else:
            imarrayi = imarray
        imarray_planefit = xyplanefit(imarrayi, polyx, polyy)  # apply plane fit
        imnorm, minval, datarange = normalise(imarray_planefit)

        imarray_planefit_list.append(imarray_planefit)
        minvallist.append(minval)
        datarangelist.append(datarange)

        h, w = imarrayi.shape

        # create mirrored array
        mirror_lr = np.concatenate((imnorm, np.fliplr(imnorm)), axis=1)
        mirror_all = np.concatenate((mirror_lr, np.flipud(mirror_lr)), axis=0)

        if imarrayi.shape == (256, 256):
            arrays = [imnorm]

        elif imarrayi.shape == (512, 512):
            arrays = [
                imnorm[0::2, 0::2],
                imnorm[0::2, 1::2],
                imnorm[1::2, 0::2],
                imnorm[1::2, 1::2],
            ]

        else:  # elif h % 256  != 0 or w % 256 != 0:

            # create downsampled 256x256 arrays
            hr = 256 * math.ceil(
                h / 256
            )  # height rounded up to nearest multiple of 256
            wr = 256 * math.ceil(w / 256)  # width rounded up to nearest multiple of 256
            hj = int(hr / 256)  # height pixel jump
            wj = int(wr / 256)  # width pixel jump

            arrays = []
            for i in range(hj):
                for j in range(wj):
                    array = mirror_all[i:hr:hj, j:wr:wj]
                    array = array[:256, :256]
                    arrays.append(
                        array
                    )  # now have arrays for each image in the stack rather than just 1 image

        arrayslist.append(arrays)

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

    def predict_on_image(model, arrays, device):
        model.eval()
        BGwindows = []
        with torch.no_grad():
            for image in arrays:
                image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                BGwindow = model(image_tensor)
                BGwindows.append(BGwindow.squeeze(0).cpu())
        return BGwindows

    # apply model to the pixelsplit arrays of each image

    predictedBGlist = []
    for arrays, minval, datarange in zip(
        arrayslist, minvallist, datarangelist, strict=False
    ):
        BGarrays = predict_on_image(loaded_model, arrays, device)  # shape [1, 256, 256]
        BGarrays_2D = [array.squeeze(0) for array in BGarrays]  # shape [256, 256]
        BGarrays_np = [array.numpy() for array in BGarrays_2D]

        if originaldim == (256, 256):
            BGstitched = BGarrays_np[0]
        elif originaldim == (512, 512):
            BGstitched = np.zeros((h, w), dtype=BGarrays_np[0].dtype)
            # Place the BG pixels back in their original positions
            BGstitched[0::2, 0::2] = BGarrays_np[0]  # Top-left positions
            BGstitched[0::2, 1::2] = BGarrays_np[1]  # Top-right positions
            BGstitched[1::2, 0::2] = BGarrays_np[2]  # Bottom-left positions
            BGstitched[1::2, 1::2] = BGarrays_np[3]  # Bottom-right positions
        else:
            BGstitched = np.zeros_like(mirror_all[:hr, :wr])
            for idx, array in enumerate(BGarrays_np):
                i = idx // wj  # row index in the sampling grid
                j = idx % wj  # column index in the sampling grid
                BGstitched[i:hr:hj, j:wr:wj] = array

            # Crop back to original dimensions
            BGstitched = BGstitched[:h, :w]

        BGwindows_denorm = denormalise(BGstitched, minval, datarange)

        predictedBG_linefit = linefit(BGwindows_denorm, lineorder)

        predictedBGlist.append(predictedBG_linefit)

    predictedLevlist = []
    for i in range(lenstack):
        predictedLev = imarray_planefit_list[i] - predictedBGlist[i]
        # Subtract median from each predictedLev to align heights across the stack
        median = np.median(predictedLev)
        predictedLev_m = predictedLev - median
        predictedLevlist.append(predictedLev_m)

    if background is True:
        return np.stack(predictedBGlist, axis=0)
    else:
        return np.stack(predictedLevlist, axis=0)
