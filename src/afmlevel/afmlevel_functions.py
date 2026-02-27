# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 13:10:12 2026

@author: pymte
"""


import math
import os
from typing import Any, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import generate_binary_structure, label
from pnanolocz_lib.level import apply_level

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~UTILS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
            y2[i, :] = np.polyval(p, x1)
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def applymodel_mask(imarray, model_path):

    dim = (256, 256)  # dimensions for image resize
    polyx = 1
    polyy = 1  # order of polynomial plane fit to initially apply to raw image
    n_channels = 1
    # won't be changing which model is used once we have found the best one, so this is defined in the function

    # ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filtersize1 = 7
    filtersize = 7  # input filter sizes used in the training !!
    leakyrelu = False
    dropoutprob = 0

    # preprocess the image
    imarray = xyplanefit(imarray, polyx, polyy)  # apply plane fit
    imarray_r = cv2.resize(imarray, dim, interpolation=cv2.INTER_NEAREST)
    imnorm, minval, maxval = normalise(imarray_r)  # normalise to values between 0 and 1

    # define model structure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    class UNet(nn.Module):
        def __init__(self, n_channels):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            padding1 = (filtersize1 - 1) // 2
            padding = (filtersize - 1) // 2

            # UNet uses a series of double convolutions (conv -> ReLU -> conv -> ReLU)
            # Batch normalisation added after convolution
            activation = (
                nn.LeakyReLU(inplace=True) if leakyrelu else nn.ReLU(inplace=True)
            )

            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(out_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            def double_conv1(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize1, padding=padding1),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(
                        out_channels, out_channels, filtersize1, padding=padding1
                    ),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            self.dc1 = double_conv1(n_channels, 16)
            self.dc2 = double_conv(16, 32)
            self.dc3 = double_conv(32, 64)
            self.dc4 = double_conv(64, 128)
            self.dc5 = double_conv(128, 256)
            self.dc6 = double_conv(256, 512)
            self.dc7 = double_conv(512, 1024)

            # Upsampling path (transposed convolutions to enlarge the inputs)
            self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dc8 = double_conv(1024, 512)
            self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dc9 = double_conv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dc10 = double_conv(256, 128)
            self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dc11 = double_conv(128, 64)
            self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dc12 = double_conv(64, 32)
            self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dc13 = double_conv(32, 16)

            # Final convolution to produce the output with desired number of classes
            self.final = nn.Conv2d(
                16, 1, 1
            )  # go from 64 channels to 1 channel, kernel size 1

            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            # Downsampling path: Repeated (double_conv -> max_pool)
            # This gradually reduces spatial dimensions while increasing feature channels
            x1 = self.dc1(x)
            x2 = self.dc2(self.max_pool(x1))
            x3 = self.dc3(self.max_pool(x2))
            x4 = self.dc4(self.max_pool(x3))
            x5 = self.dc5(self.max_pool(x4))
            x6 = self.dc6(self.max_pool(x5))
            x7 = self.dc7(self.max_pool(x6))

            # Upsampling with skip connections
            # Each step: upconv -> concatenate with corresponding downsampling layer -> double_conv
            x = self.up1(x7)
            x = self.dc8(torch.cat([x6, x], dim=1))
            x = self.up2(x)
            x = self.dc9(torch.cat([x5, x], dim=1))
            x = self.up3(x)
            x = self.dc10(torch.cat([x4, x], dim=1))
            x = self.up4(x)
            x = self.dc11(torch.cat([x3, x], dim=1))
            x = self.up5(x)
            x = self.dc12(torch.cat([x2, x], dim=1))
            x = self.up6(x)
            x = self.dc13(torch.cat([x1, x], dim=1))

            return self.final(x)

    def load_model(model_path, n_channels, device):
        model = UNet(n_channels=n_channels)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
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

    # ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filtersize1 = 7
    filtersize = 7  # input filter sizes used in the training !!
    leakyrelu = False
    dropoutprob = 0

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

    # define model structure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    class UNet(nn.Module):
        def __init__(self, n_channels):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            padding1 = (filtersize1 - 1) // 2
            padding = (filtersize - 1) // 2

            # UNet uses a series of double convolutions (conv -> ReLU -> conv -> ReLU)
            # Batch normalisation added after convolution
            activation = (
                nn.LeakyReLU(inplace=True) if leakyrelu else nn.ReLU(inplace=True)
            )

            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(out_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            def double_conv1(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize1, padding=padding1),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(
                        out_channels, out_channels, filtersize1, padding=padding1
                    ),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            self.dc1 = double_conv1(n_channels, 16)
            self.dc2 = double_conv(16, 32)
            self.dc3 = double_conv(32, 64)
            self.dc4 = double_conv(64, 128)
            self.dc5 = double_conv(128, 256)
            self.dc6 = double_conv(256, 512)
            self.dc7 = double_conv(512, 1024)

            # Upsampling path (transposed convolutions to enlarge the inputs)
            self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dc8 = double_conv(1024, 512)
            self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dc9 = double_conv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dc10 = double_conv(256, 128)
            self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dc11 = double_conv(128, 64)
            self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dc12 = double_conv(64, 32)
            self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dc13 = double_conv(32, 16)

            # Final convolution to produce the output with desired number of classes
            self.final = nn.Conv2d(
                16, 1, 1
            )  # go from 64 channels to 1 channel, kernel size 1

            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            # Downsampling path: Repeated (double_conv -> max_pool)
            # This gradually reduces spatial dimensions while increasing feature channels
            x1 = self.dc1(x)
            x2 = self.dc2(self.max_pool(x1))
            x3 = self.dc3(self.max_pool(x2))
            x4 = self.dc4(self.max_pool(x3))
            x5 = self.dc5(self.max_pool(x4))
            x6 = self.dc6(self.max_pool(x5))
            x7 = self.dc7(self.max_pool(x6))

            # Upsampling with skip connections
            # Each step: upconv -> concatenate with corresponding downsampling layer -> double_conv
            x = self.up1(x7)
            x = self.dc8(torch.cat([x6, x], dim=1))
            x = self.up2(x)
            x = self.dc9(torch.cat([x5, x], dim=1))
            x = self.up3(x)
            x = self.dc10(torch.cat([x4, x], dim=1))
            x = self.up4(x)
            x = self.dc11(torch.cat([x3, x], dim=1))
            x = self.up5(x)
            x = self.dc12(torch.cat([x2, x], dim=1))
            x = self.up6(x)
            x = self.dc13(torch.cat([x1, x], dim=1))

            return self.final(x)

    def load_model(model_path, n_channels, device):
        model = UNet(n_channels=n_channels)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
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

    # ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filtersize1 = 9
    filtersize = 9  # input filter sizes used in the training !!
    leakyrelu = False
    dropoutprob = 0

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

    # define model structure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    class UNet(nn.Module):
        def __init__(self, n_channels):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            padding1 = (filtersize1 - 1) // 2
            padding = (filtersize - 1) // 2

            # UNet uses a series of double convolutions (conv -> ReLU -> conv -> ReLU)
            # Batch normalisation added after convolution
            activation = (
                nn.LeakyReLU(inplace=True) if leakyrelu else nn.ReLU(inplace=True)
            )

            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(out_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            def double_conv1(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize1, padding=padding1),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(
                        out_channels, out_channels, filtersize1, padding=padding1
                    ),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            self.dc1 = double_conv1(n_channels, 16)
            self.dc2 = double_conv(16, 32)
            self.dc3 = double_conv(32, 64)
            self.dc4 = double_conv(64, 128)
            self.dc5 = double_conv(128, 256)
            self.dc6 = double_conv(256, 512)
            self.dc7 = double_conv(512, 1024)

            # Upsampling path (transposed convolutions to enlarge the inputs)
            self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dc8 = double_conv(1024, 512)
            self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dc9 = double_conv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dc10 = double_conv(256, 128)
            self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dc11 = double_conv(128, 64)
            self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dc12 = double_conv(64, 32)
            self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dc13 = double_conv(32, 16)

            # Final convolution to produce the output with desired number of classes
            self.final = nn.Conv2d(
                16, 1, 1
            )  # go from 64 channels to 1 channel, kernel size 1

            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            # Downsampling path: Repeated (double_conv -> max_pool)
            # This gradually reduces spatial dimensions while increasing feature channels
            x1 = self.dc1(x)
            x2 = self.dc2(self.max_pool(x1))
            x3 = self.dc3(self.max_pool(x2))
            x4 = self.dc4(self.max_pool(x3))
            x5 = self.dc5(self.max_pool(x4))
            x6 = self.dc6(self.max_pool(x5))
            x7 = self.dc7(self.max_pool(x6))

            # Upsampling with skip connections
            # Each step: upconv -> concatenate with corresponding downsampling layer -> double_conv
            x = self.up1(x7)
            x = self.dc8(torch.cat([x6, x], dim=1))
            x = self.up2(x)
            x = self.dc9(torch.cat([x5, x], dim=1))
            x = self.up3(x)
            x = self.dc10(torch.cat([x4, x], dim=1))
            x = self.up4(x)
            x = self.dc11(torch.cat([x3, x], dim=1))
            x = self.up5(x)
            x = self.dc12(torch.cat([x2, x], dim=1))
            x = self.up6(x)
            x = self.dc13(torch.cat([x1, x], dim=1))

            return self.final(x)

    def load_model(model_path: str, n_channels: int, device: str):
        model = UNet(n_channels=n_channels)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
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

    # Ensure imarray is 3D (N, H, W) for consistent processing
    if imarray.ndim == 2:
        imarray = imarray[np.newaxis, :, :]
        was_2d = True
    elif imarray.ndim == 3:
        was_2d = False
    else:
        raise ValueError(
            "imarray must be either 2D or 3D with shape (H, W) or (N, H, W)"
        )

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

    # ~~~~~~~~~~~~~~~~~~~~~MODEL SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    filtersize1 = 9
    filtersize = 9  # input filter sizes used in the training !!
    leakyrelu = False
    dropoutprob = 0

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

    # define model structure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    class UNet(nn.Module):
        def __init__(self, n_channels):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            padding1 = (filtersize1 - 1) // 2
            padding = (filtersize - 1) // 2

            # UNet uses a series of double convolutions (conv -> ReLU -> conv -> ReLU)
            # Batch normalisation added after convolution
            activation = (
                nn.LeakyReLU(inplace=True) if leakyrelu else nn.ReLU(inplace=True)
            )

            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(out_channels, out_channels, filtersize, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            def double_conv1(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, filtersize1, padding=padding1),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                    nn.Conv2d(
                        out_channels, out_channels, filtersize1, padding=padding1
                    ),
                    nn.BatchNorm2d(out_channels),
                    activation,
                    nn.Dropout(dropoutprob),
                )

            self.dc1 = double_conv1(n_channels, 16)
            self.dc2 = double_conv(16, 32)
            self.dc3 = double_conv(32, 64)
            self.dc4 = double_conv(64, 128)
            self.dc5 = double_conv(128, 256)
            self.dc6 = double_conv(256, 512)
            self.dc7 = double_conv(512, 1024)

            # Upsampling path (transposed convolutions to enlarge the inputs)
            self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dc8 = double_conv(1024, 512)
            self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dc9 = double_conv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dc10 = double_conv(256, 128)
            self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dc11 = double_conv(128, 64)
            self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dc12 = double_conv(64, 32)
            self.up6 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dc13 = double_conv(32, 16)

            # Final convolution to produce the output with desired number of classes
            self.final = nn.Conv2d(
                16, 1, 1
            )  # go from 64 channels to 1 channel, kernel size 1

            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            # Downsampling path: Repeated (double_conv -> max_pool)
            # This gradually reduces spatial dimensions while increasing feature channels
            x1 = self.dc1(x)
            x2 = self.dc2(self.max_pool(x1))
            x3 = self.dc3(self.max_pool(x2))
            x4 = self.dc4(self.max_pool(x3))
            x5 = self.dc5(self.max_pool(x4))
            x6 = self.dc6(self.max_pool(x5))
            x7 = self.dc7(self.max_pool(x6))

            # Upsampling with skip connections
            # Each step: upconv -> concatenate with corresponding downsampling layer -> double_conv
            x = self.up1(x7)
            x = self.dc8(torch.cat([x6, x], dim=1))
            x = self.up2(x)
            x = self.dc9(torch.cat([x5, x], dim=1))
            x = self.up3(x)
            x = self.dc10(torch.cat([x4, x], dim=1))
            x = self.up4(x)
            x = self.dc11(torch.cat([x3, x], dim=1))
            x = self.up5(x)
            x = self.dc12(torch.cat([x2, x], dim=1))
            x = self.up6(x)
            x = self.dc13(torch.cat([x1, x], dim=1))

            return self.final(x)

    def load_model(model_path, n_channels, device):
        model = UNet(n_channels=n_channels)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
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
        return predictedBGlist
    else:
        return predictedLevlist
