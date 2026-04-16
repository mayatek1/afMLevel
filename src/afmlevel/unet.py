"""
U-Net model structure definition and loading utility for afMLevel.

This module defines the UNet class for the mask and background models and provides a
unified function to load the model with caching to optimise performance when processing
multiple images or stacks.

AI Transparency Note
--------------------
AI-based tools were used in certain parts of this module for limited typing/formatting
assistance and for providing debugging, refactoring and documentation suggestions. All
code paths, algorithms, and final behaviour were reviewed and validated by the authors.
"""

import os

import torch
import torch.nn as nn
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)
# Global model cache
_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def load_unet_model(
    model_path: str,
    n_channels: int,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """
    Load a U-Net model from either a local .pth file or a Hugging Face Hub repository.

    Parameters
    ----------
    model_path : str
        Path or identifier for the model. Supported formats:
        - Local file path: "path/to/model.pth"
        - HuggingFace repo: "user/repo" (loads "model.pth")
        - HuggingFace repo with explicit file: "user/repo::file.pth"
    n_channels : int
        Number of input channels (usually 1)
    config : dict
        Dictionary of U-Net configuration parameters.
    device : torch.device
        Device on which to load the model. Currently: 'cuda' or 'cpu'

    Returns
    -------
    nn.Module
        Model loaded on the requested device.

    Notes
    -----
    - Models from HuggingFace are cached locally and reused automatically.
    - The function maintains an internal cache keyed by (resolved_path, device)
      to avoid reloading models repeatedly within a session.
    """
    # Check HF format:  "repo::filename"
    if "::" in model_path:
        repo_id, filename = model_path.split("::", 1)
        hf_file = hf_hub_download(repo_id=repo_id, filename=filename)
        real_path = hf_file

    # If model_path has no extension, treat as HF repo with default filename
    elif not os.path.exists(model_path):
        # assume repo_id and default filename "model.pth"
        repo_id = model_path
        hf_file = hf_hub_download(repo_id=repo_id, filename="model.pth")
        real_path = hf_file

    else:
        # local file
        real_path = model_path

    key = (os.path.abspath(real_path), str(device))

    if key in _MODEL_CACHE:
        logger.debug("Reusing cached UNet for key=%s", key)
        return _MODEL_CACHE[key]

    logger.info(f"Loading UNet weights from: {real_path}")
    logger.info(
        "Loading UNet (n_channels=%s, device=%s, config=%s)",
        n_channels,
        device,
        config,
    )
    model = UNet(n_channels, **config).to(device)

    # Always load from the TRUE resolved file path (real_path)
    try:
        state = torch.load(real_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(real_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _MODEL_CACHE[key] = model
    logger.info(f"Model loaded & cached from: {real_path}")
    return model


class UNet(nn.Module):
    def __init__(
        self, n_channels, filtersize1=9, filtersize=9, leakyrelu=False, dropoutprob=0
    ):
        super().__init__()

        padding1 = (filtersize1 - 1) // 2
        padding = (filtersize - 1) // 2

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

        self.dc1 = double_conv1(n_channels, 16)
        self.dc2 = double_conv(16, 32)
        self.dc3 = double_conv(32, 64)
        self.dc4 = double_conv(64, 128)
        self.dc5 = double_conv(128, 256)
        self.dc6 = double_conv(256, 512)
        self.dc7 = double_conv(512, 1024)

        # Upsampling
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

        self.final = nn.Conv2d(16, 1, 1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(self.max_pool(x1))
        x3 = self.dc3(self.max_pool(x2))
        x4 = self.dc4(self.max_pool(x3))
        x5 = self.dc5(self.max_pool(x4))
        x6 = self.dc6(self.max_pool(x5))
        x7 = self.dc7(self.max_pool(x6))

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
