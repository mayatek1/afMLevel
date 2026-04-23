"""Pytest fixtures for testing the UNet model in afmlevel.unet."""

import numpy as np
import pytest

from afmlevel.unet import UNet


@pytest.fixture
def tiny_unet_cpu():
    """Return a minimal UNet instance configured for fast CPU-based testing."""
    # filtersize must be odd; use 3 for speed in tests
    model = UNet(
        n_channels=1,
        filtersize1=3,
        filtersize=3,
        leakyrelu=False,
        dropoutprob=0,
    )
    model.eval()
    return model


@pytest.fixture
def dummy_256_image():
    """Provide a random 256x256 float32 image for model input tests."""
    return np.random.rand(256, 256).astype(np.float32)
