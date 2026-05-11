"""Shared pytest fixtures for afmlevel tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from afmlevel.unet import _MODEL_CACHE, UNet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).parent
RESOURCES_DIR = TESTS_DIR / "resources"

# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------
# Expected outputs for the committed sample TIFFs.
# Update these deliberately when model weights change — don't just fix failing tests.
# Last updated: 2026-04-27
# Model weights: background_unet.pth, mask_unet.pth
# Weights source:
# * mask_unet: https://huggingface.co/Heath-AFM-Lab/afMLevel-mask-unet (commit 27dd728) # noqa
# * background_unet: https://huggingface.co/Heath-AFM-Lab/afMLevel-background-unet (commit ff669bc) # noqa
# Sample data: tests/resources/sample_2d.tiff (200×200), sample_3d.tiff (3×256×256)
# ---------------------------------------------------------------------------

# 2D image — level_ml_bg output
_EXPECTED_2D_BG_MEAN: float | None = -0.319597  # nm
_EXPECTED_2D_BG_STD: float | None = 1.009109  # nm

# 3D stack — level_ml_bg output, per-slice means after zero_median
_EXPECTED_3D_BG_SLICE_MEANS: list[float] | None = [-1.835143, -1.826935, -1.841088]
_EXPECTED_3D_BG_SLICE_STDS: list[float] | None = [2.363965, 2.368027, 2.342659]

# 2D image — ml_mask output
_EXPECTED_2D_MASK_FOREGROUND_FRACTION: float | None = 0.542075

# 2D image — level_ml_mask output
_EXPECTED_2D_MASKED_MEAN: float | None = -0.835184  # nm
_EXPECTED_2D_MASKED_STD: float | None = 0.986389  # nm

# ---------------------------------------------------------------------------
# Fast fixtures — tiny mock model (used by existing tests, unchanged)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_unet_cpu():
    """Return a minimal UNet instance configured for fast CPU-based testing."""
    # filtersize must be odd; use 3 for speed in tests
    model = UNet(
        n_channels=1,
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


# ---------------------------------------------------------------------------
# Fast fixtures — mock patchers (convenience wrappers used in multiple files)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_bg_model(tiny_unet_cpu):
    """Patch load_unet_model in background_model with the tiny CPU UNet."""
    _MODEL_CACHE.clear()
    with patch(
        "afmlevel.background_model.load_unet_model",
        return_value=tiny_unet_cpu,
    ):
        yield
    _MODEL_CACHE.clear()


@pytest.fixture
def mock_mask_model(tiny_unet_cpu):
    """Patch load_unet_model in mask_model with the tiny CPU UNet."""
    _MODEL_CACHE.clear()
    with patch(
        "afmlevel.mask_model.load_unet_model",
        return_value=tiny_unet_cpu,
    ):
        yield
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Slow fixtures — real TIFFs from tests/resources/
# ---------------------------------------------------------------------------


def _discover_tiffs() -> tuple[Path | None, Path | None]:
    """Scan resources/ and return (first_2d_tiff, first_3d_tiff)."""
    found_2d: Path | None = None
    found_3d: Path | None = None
    for p in sorted(RESOURCES_DIR.glob("*.tif*")):
        try:
            arr = tifffile.imread(str(p))
        except Exception:
            continue
        if arr.ndim == 2 and found_2d is None:
            found_2d = p
        elif arr.ndim == 3 and found_3d is None:
            found_3d = p
        if found_2d and found_3d:
            break
    return found_2d, found_3d


# Session-scoped so the file is read once regardless of how many slow tests use it
@pytest.fixture(scope="session")
def real_image_2d() -> np.ndarray:
    """Load a real 2D AFM image from tests/resources/."""
    path_2d, _ = _discover_tiffs()
    if path_2d is None:
        pytest.skip("No 2D TIFF found in tests/resources/ — skipping slow test.")
    return tifffile.imread(str(path_2d)).astype(np.float64)


@pytest.fixture(scope="session")
def real_image_3d() -> np.ndarray:
    """Load a real 3D AFM stack from tests/resources/."""
    _, path_3d = _discover_tiffs()
    if path_3d is None:
        pytest.skip("No 3D TIFF found in tests/resources/ — skipping slow test.")
    return tifffile.imread(str(path_3d)).astype(np.float64)
