# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 13:31:29 2025

@author: pymte

Mask Model Training on train dataset 3 & 4 - Simplified to work on Aire Only - Variable params
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.transform import resize
import pandas as pd


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


def med_line(imarray, background_preservation=True):
    row_medians = np.median(imarray, axis=1, keepdims=True)
    corrected_imarray = imarray - row_medians

    if background_preservation:
        global_median = np.median(imarray)
        corrected_imarray += global_median

    return corrected_imarray


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalise(image):
    if isinstance(image, torch.Tensor):
        im0 = image - image.min()
        imnorm = im0 / (im0.max() + 1e-8)
        return imnorm
    elif isinstance(image, np.ndarray):
        im0 = image - np.min(image)
        imnorm = im0 / (np.max(im0) + 1e-8)
        return imnorm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~DEFINE VARIABLES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
displayfigs = True
leakyrelu = False
LR = "_LR" if leakyrelu else ""
dropoutprob = 0.0
dropoutprobl = int(dropoutprob * 100)

testrun = False  # input True to run using a small portion of the training data
dim = (256, 256)  # dimensions for image resize
polyx = 1  # order of polynomial plane fit to initially apply to raw image
polyy = 1
linefit = False
LF = "_LF" if linefit else ""

num_lineimages = (
    2  # number of line images to add to augmented training data. Try 0, 2, 4
)
num_epochs = 80
batch_size = 32  # try 4, 32, 64
otsu_sigma = 1  # gaussian blur for generation of masks by otsu

num_layers = 7  # Options: 3, 5, 7, 9 only
filtersize1 = 7  # any odd number > 1. Recommend trying 3, 5, 7, 9
filtersize = 7

n_saved = 15  # save model at each of the last n epochs
extralabel = f"_train34_otsublur{otsu_sigma}{LF}"
label = f"Aire_Mask_{num_epochs}eps_b{batch_size}_d{dropoutprobl}_1f{filtersize1}_f{filtersize}_{num_layers}l{LR}_line{num_lineimages}{extralabel}"

# 2) load the data
base_path = os.getcwd()
train_data = []
image_path1 = os.path.join(base_path, "Image")
levelled_path1 = os.path.join(base_path, "Levelled")

# make a list of image ids
image_ids = []

for filename in os.listdir(image_path1):
    base, ext = os.path.splitext(filename)
    image_ids.append((base, ext))

print("Number of images:", len(image_ids))

# format the training images and groundtruths into a dictionary
for image_id, ext in image_ids:

    image_file = f"{image_id}{ext}"
    levelled_file = f"{image_id}_levelled{ext}"

    image_path = os.path.join(image_path1, image_file)
    levelled_path = os.path.join(levelled_path1, levelled_file)

    image = Image.open(image_path)  # open image
    imr = image.resize(
        dim, Image.NEAREST
    )  # resize to 512x512 pix (nearest neighbour interp.)
    imarray = np.array(imr)  # convert to array
    imarray = xyplanefit(imarray, polyx, polyy)  # apply plane fit
    if linefit:
        imarray = med_line(imarray)
    imnorm = normalise(imarray)  # normalise to values between 0 and 1  OR

    levelled = Image.open(levelled_path)  # open levelled image
    levr = levelled.resize(
        dim, Image.NEAREST
    )  # resize to 512x512 pix (nearest neighbour interp.)
    levarray = np.array(levr)  # convert to array
    levnorm = normalise(levarray)  # normalise to values between 0 and 1

    # GENERATE MASK FROM LEVELLED IMAGE
    mean_value = np.mean(levarray)
    levshift = levarray - mean_value
    maxval = np.max(levarray)

    slev = sigmoid(levshift)
    slev8 = (slev * 255).astype(np.uint8)  # Scale sigmoid output to 0-255 range

    # Otsu's thresholding
    blur = gaussian_filter(slev8, sigma=otsu_sigma)
    thresh = threshold_otsu(blur)
    th3 = (blur > thresh).astype(np.uint8) * 255
    maskr = resize(th3, dim, order=0, preserve_range=True, anti_aliasing=False).astype(
        np.uint8
    )
    maskr = maskr / 255.0

    train_data.append({"image_id": image_id, "image": imnorm, "groundtruth": maskr})


# split into a training and validation set
train_data, val_data = train_test_split(
    train_data, train_size=0.8, shuffle=True, random_state=42
)
print(f"Initial number of validation samples: {len(val_data)}")
print(f"Initial number of training samples: {len(train_data)}")


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        padding1 = (filtersize1 - 1) // 2
        padding = (filtersize - 1) // 2

        # UNet uses a series of double convolutions (conv -> ReLU -> conv -> ReLU)
        # Batch normalisation added after convolution
        activation = nn.LeakyReLU(inplace=True) if leakyrelu else nn.ReLU(inplace=True)

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
                nn.Conv2d(out_channels, out_channels, filtersize1, padding=padding1),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Dropout(dropoutprob),
            )

        if num_layers == 3:
            self.dc1 = double_conv1(n_channels, 64)
            self.dc2 = double_conv(64, 128)
            self.dc3 = double_conv(128, 256)

            # Upsampling path (transposed convolutions to enlarge the inputs)
            self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dc4 = double_conv(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dc5 = double_conv(128, 64)

            # Final convolution to produce the output with desired number of classes
            self.final = nn.Conv2d(
                64, 1, 1
            )  # go from 64 channels to 1 channel, kernel size 1
            self.max_pool = nn.MaxPool2d(2)

        elif num_layers == 5:
            self.dc1 = double_conv1(n_channels, 64)
            self.dc2 = double_conv(64, 128)
            self.dc3 = double_conv(128, 256)
            self.dc4 = double_conv(256, 512)
            self.dc5 = double_conv(512, 1024)

            # Upsampling path (transposed convolutions to enlarge the inputs)
            self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dc6 = double_conv(1024, 512)
            self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dc7 = double_conv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dc8 = double_conv(256, 128)
            self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dc9 = double_conv(128, 64)

            # Final convolution to produce the output with desired number of classes
            self.final = nn.Conv2d(
                64, 1, 1
            )  # go from 64 channels to 1 channel, kernel size 1
            self.max_pool = nn.MaxPool2d(2)

        elif num_layers == 7:
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

        elif num_layers == 9:
            self.dc1 = double_conv1(n_channels, 8)
            self.dc2 = double_conv(8, 16)
            self.dc3 = double_conv(16, 32)
            self.dc4 = double_conv(32, 64)
            self.dc5 = double_conv(64, 128)
            self.dc6 = double_conv(128, 256)
            self.dc7 = double_conv(256, 512)
            self.dc8 = double_conv(512, 1024)
            self.dc9 = double_conv(1024, 2048)

            # Upsampling path (transposed convolutions to enlarge the inputs)
            self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
            self.dc10 = double_conv(2048, 1024)
            self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dc11 = double_conv(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dc12 = double_conv(512, 256)
            self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dc13 = double_conv(256, 128)
            self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dc14 = double_conv(128, 64)
            self.up6 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dc15 = double_conv(64, 32)
            self.up7 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dc16 = double_conv(32, 16)
            self.up8 = nn.ConvTranspose2d(16, 8, 2, stride=2)
            self.dc17 = double_conv(16, 8)

            # Final convolution to produce the output with desired number of classes
            self.final = nn.Conv2d(
                8, 1, 1
            )  # go from 64 channels to 1 channel, kernel size 1
            self.max_pool = nn.MaxPool2d(2)

    if num_layers == 3:

        def forward(self, x):
            # Downsampling path: Repeated (double_conv -> max_pool)
            # This gradually reduces spatial dimensions while increasing feature channels
            x1 = self.dc1(x)
            x2 = self.dc2(self.max_pool(x1))
            x3 = self.dc3(self.max_pool(x2))

            # Upsampling with skip connections
            # Each step: upconv -> concatenate with corresponding downsampling layer -> double_conv
            x = self.up1(x3)
            x = self.dc4(torch.cat([x2, x], dim=1))
            x = self.up2(x)
            x = self.dc5(torch.cat([x1, x], dim=1))

            return self.final(x)

    elif num_layers == 5:

        def forward(self, x):
            # Downsampling path: Repeated (double_conv -> max_pool)
            # This gradually reduces spatial dimensions while increasing feature channels
            x1 = self.dc1(x)
            x2 = self.dc2(self.max_pool(x1))
            x3 = self.dc3(self.max_pool(x2))
            x4 = self.dc4(self.max_pool(x3))
            x5 = self.dc5(self.max_pool(x4))

            # Upsampling with skip connections
            # Each step: upconv -> concatenate with corresponding downsampling layer -> double_conv
            x = self.up1(x5)
            x = self.dc6(torch.cat([x4, x], dim=1))
            x = self.up2(x)
            x = self.dc7(torch.cat([x3, x], dim=1))
            x = self.up3(x)
            x = self.dc8(torch.cat([x2, x], dim=1))
            x = self.up4(x)
            x = self.dc9(torch.cat([x1, x], dim=1))

            return self.final(x)

    elif num_layers == 7:

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

    elif num_layers == 9:

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
            x8 = self.dc8(self.max_pool(x7))
            x9 = self.dc9(self.max_pool(x8))

            # Upsampling with skip connections
            # Each step: upconv -> concatenate with corresponding downsampling layer -> double_conv
            x = self.up1(x9)
            x = self.dc10(torch.cat([x8, x], dim=1))
            x = self.up2(x)
            x = self.dc11(torch.cat([x7, x], dim=1))
            x = self.up3(x)
            x = self.dc12(torch.cat([x6, x], dim=1))
            x = self.up4(x)
            x = self.dc13(torch.cat([x5, x], dim=1))
            x = self.up5(x)
            x = self.dc14(torch.cat([x4, x], dim=1))
            x = self.up6(x)
            x = self.dc15(torch.cat([x3, x], dim=1))
            x = self.up7(x)
            x = self.dc16(torch.cat([x2, x], dim=1))
            x = self.up8(x)
            x = self.dc17(torch.cat([x1, x], dim=1))

            return self.final(x)


def artificial_linenoise(image, n):
    # Generate n images with random line noise for each image
    line_images = []

    for _ in range(n):
        line_image = image.copy()
        nlines = random.randint(1, 3)

        for _ in range(nlines):
            width = random.randint(5, 50)
            height = random.uniform(np.min(image), np.max(image) / 4)
            start_row = random.randint(0, image.shape[0] - width)

            linearray = np.zeros_like(image)
            linearray[start_row : start_row + width, :] = height

            if random.choice([True, False]):
                line_image += linearray
            else:
                line_image -= linearray

            lineim0 = line_image - np.min(
                line_image
            )  # normalise the line image to values between 0 and 1
            line_image = lineim0 / np.max(lineim0)

        line_images.append(line_image)

    return line_images


def rotate_image_and_groundtruth(image, groundtruth, angle):
    rotated_image = np.ascontiguousarray(np.rot90(image, k=angle // 90))
    rotated_groundtruth = np.ascontiguousarray(np.rot90(groundtruth, k=angle // 90))
    return rotated_image, rotated_groundtruth


def flip_image_and_groundtruth(image, groundtruth, flip_type):
    if flip_type == "horizontal":
        flipped_image = np.ascontiguousarray(np.fliplr(image))
        flipped_groundtruth = np.ascontiguousarray(np.fliplr(groundtruth))
    elif flip_type == "vertical":
        flipped_image = np.ascontiguousarray(np.flipud(image))
        flipped_groundtruth = np.ascontiguousarray(np.flipud(groundtruth))
    else:
        return image, groundtruth
    return flipped_image, flipped_groundtruth


def augment_data(data):
    augmented_data = []
    for item in data:
        image = item["image"]
        groundtruth = item["groundtruth"]

        # Original image
        augmented_data.append(item)

        # Rotate 180 degrees
        rotated_image, rotated_groundtruth = rotate_image_and_groundtruth(
            image, groundtruth, 180
        )
        augmented_data.append(
            {
                "image_id": f"{item['image_id']}_rotated180",
                "image": rotated_image,
                "groundtruth": rotated_groundtruth,
            }
        )

        line_images = artificial_linenoise(image, num_lineimages)
        for n in range(len(line_images)):
            augmented_data.append(
                {
                    "image_id": f"{item['image_id']}_linenoise{n}",
                    "image": line_images[n],
                    "groundtruth": groundtruth,
                }
            )

        # Flip horizontally
        flipped_image, flipped_groundtruth = flip_image_and_groundtruth(
            image, groundtruth, "horizontal"
        )
        augmented_data.append(
            {
                "image_id": f"{item['image_id']}_flipped_h",
                "image": flipped_image,
                "groundtruth": flipped_groundtruth,
            }
        )

    return augmented_data


# Replace the original train_data and val_data with the augmented version
train_data = augment_data(train_data)
print(f"Number of training samples after augmentation: {len(train_data)}")
val_data = augment_data(val_data)
print(f"Number of validation samples after augmentation: {len(val_data)}")


class LevellingDataset(Dataset):
    def __init__(self, data, transform=None):  # , max_instances=20):
        # Initialize the dataset with raw data, optional transforms, and max instances
        self.data = data
        self.transform = transform

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a single sample from the dataset
        sample = self.data[idx]
        image = sample["image"]
        groundtruth = sample["groundtruth"]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        groundtruth_tensor = torch.tensor(groundtruth, dtype=torch.float32).unsqueeze(0)
        instance_groundtruth = groundtruth_tensor

        return image, instance_groundtruth


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )
    return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss_val = dice_loss(torch.sigmoid(pred), target)
        return bce_loss + dice_loss_val


# This function will allow to visualize predicted vs ground truth segmentation masks/ levelled images during training
def visualize_instance_groundtruths(
    groundtruth, prediction, sprediction, bprediction, image, bimage, batch_idx, epoch
):
    # Convert image tensor to numpy array and transpose to (H, W, C)
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    prediction_np = prediction.cpu().numpy().transpose(1, 2, 0)
    sprediction_np = sprediction.cpu().numpy().transpose(1, 2, 0)
    bprediction_np = bprediction.cpu().numpy().transpose(1, 2, 0)
    groundtruth_np = groundtruth.cpu().numpy().transpose(1, 2, 0)

    # Create two separate subplots
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    ax = axes.ravel()

    # Top subplot: Original layout
    im0 = ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(groundtruth_np)  # , vmin = 0, vmax = 1)
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(prediction_np)
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2])

    im3 = ax[3].imshow(sprediction_np)
    ax[3].set_title("Sigmoid Predicted Mask")
    ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3])

    im4 = ax[4].imshow(bprediction_np, vmin=0, vmax=1)
    ax[4].set_title("Binarised Predicted Mask")
    ax[4].axis("off")
    fig.colorbar(im4, ax=ax[4])

    # Set the overall title
    fig.suptitle(f"Epoch {epoch+1}, Batch {batch_idx+1}", fontsize=16)

    # Adjust the layout
    plt.tight_layout()

    # Close the figure
    # plt.close(fig)

    return fig


# Create datasets and dataloaders
if testrun:
    train_data = train_data[:20]  # Reduce the size of the train_data to 20
    val_data = val_data[:5]
train_dataset = LevellingDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = LevellingDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1).to(device)  # , max_instances=max_instances).to(device)
criterion = CombinedLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ====== #
trainlosslist = []
vallosslist = []
for epoch in range(num_epochs):
    print(f"====== Epoch number {epoch+1} ======")
    model.train()
    train_loss = 0
    for batch_idx, (images, groundtruths) in enumerate(train_loader):
        images = images.to(device)
        groundtruths = groundtruths.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, groundtruths.float())

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}")

    # validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (images, groundtruths) in enumerate(val_loader):
            images = images.to(device)
            groundtruths = groundtruths.to(device)
            outputs = model(images)
            loss = criterion(outputs, groundtruths.float())
            val_loss += loss.item()
            valprediction = outputs[0]
            # Visualize some validation images from final epoch
            if epoch == num_epochs - 1:
                if batch_idx % 3 == 0:
                    with torch.no_grad():
                        model.eval()
                        image = images[0]
                        bimage = (image > 0.5).float()

                        prediction = outputs[0]
                        sprediction = torch.sigmoid(
                            prediction
                        )  # sigmoid to get between 0 and 1
                        bprediction = (sprediction > 0.5).float()  # binarise

                        groundtruth = groundtruths[0]

                        fig = visualize_instance_groundtruths(
                            groundtruth,
                            prediction,
                            sprediction,
                            bprediction,
                            image,
                            bimage,
                            batch_idx,
                            epoch,
                        )
                        fig.suptitle("Validation")

                        plt.savefig(f"Validation_{label}_{batch_idx}.png")
                        plt.close(fig)
                        model.train()

    val_loss /= len(val_loader)
    vallosslist.append(val_loss)

    train_loss /= len(train_loader)
    trainlosslist.append(train_loss)

    # save the model at each of the last n epochs - then can pick the best one to download based on the loss curve
    if epoch >= num_epochs - n_saved:
        torch.save(model.state_dict(), f"unetmodel_{label}_epoch{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

    print(f"'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
    print(f"Val Loss: {val_loss:.6f}")

plt.figure()
x = np.arange(1, epoch + 2)
ytrain = trainlosslist
yval = vallosslist
plt.plot(x, ytrain, label="Training Loss", marker="o", color="plum")
plt.plot(x, yval, label="Validation Loss", marker="o", color="royalblue")
plt.title(f"Loss_{label}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.7)
plt.minorticks_on()
plt.grid(which="minor", linestyle=":", alpha=0.5)
plt.ylim(0, 1)
plt.legend()
plt.savefig(f"Lossfinal_{label}.png")
plt.close(fig)

df = pd.DataFrame({"Train Loss": trainlosslist, "Val Loss": vallosslist})
df.to_csv(f"Lossfinal_{label}.csv", index=False)

# Save the trained model
torch.save(model.state_dict(), f"unetmodel_{label}.pth")
print("Model saved")
