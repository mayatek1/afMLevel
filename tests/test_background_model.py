"""Tests for ML-based background levelling via level_ml_bg."""

from unittest.mock import patch

import numpy as np
import pytest

from afmlevel.background_model import level_ml_bg


@pytest.fixture
def mock_load_unet(tiny_unet_cpu):
    """Mock UNet model loading to avoid disk access during tests."""
    with patch(
        "afmlevel.background_model.load_unet_model",
        return_value=tiny_unet_cpu,
    ) as m:
        yield m


class TestLevelMlBg:
    """Test suite for the level_ml_bg function."""

    def test_2d_input_returns_2d(self, mock_load_unet):
        """Ensure 2D input returns a 2D output of the same shape."""
        img = np.random.rand(256, 256).astype(np.float32)
        result = level_ml_bg(img)
        assert result.ndim == 2
        assert result.shape == (256, 256)

    def test_3d_input_returns_3d(self, mock_load_unet):
        """Ensure 3D input stacks preserve dimensionality and shape."""
        stack = np.random.rand(3, 256, 256).astype(np.float32)
        result = level_ml_bg(stack)
        assert result.ndim == 3
        assert result.shape == (3, 256, 256)

    def test_background_flag_returns_background(self, mock_load_unet):
        """Verify background and levelled outputs differ when flag is set."""
        img = np.random.rand(256, 256).astype(np.float32)
        bg = level_ml_bg(img, background=True)
        lev = level_ml_bg(img, background=False)
        assert not np.allclose(bg, lev)

    def test_invalid_ndim_raises(self, mock_load_unet):
        """Confirm invalid input dimensionality raises a ValueError."""
        with pytest.raises(ValueError, match="2D or 3D"):
            level_ml_bg(np.zeros((2, 3, 4, 5)))

    def test_non_square_512x512(self, mock_load_unet):
        """Ensure a 512x512 image is processed without shape change."""
        img = np.random.rand(512, 512).astype(np.float32)
        result = level_ml_bg(img)
        assert result.shape == (512, 512)

    def test_general_size(self, mock_load_unet):
        """Ensure arbitrary image sizes are preserved after processing."""
        img = np.random.rand(300, 400).astype(np.float32)
        result = level_ml_bg(img)
        assert result.shape == (300, 400)

    def test_zero_median_true_centres_2d_output(self, mock_load_unet):
        """Test that zero_median=True produces a near-zero median for 2D output."""
        img = np.random.rand(256, 256).astype(np.float32) + 10.0  # large offset
        result = level_ml_bg(img, zero_median=True)
        assert abs(np.median(result)) < abs(np.median(img))

    def test_zero_median_false_does_not_centre_2d_output(self, mock_load_unet):
        """Test that zero_median=False does not subtract the median from 2D output."""
        img = np.random.rand(256, 256).astype(np.float32)
        result_on = level_ml_bg(img, zero_median=True)
        result_off = level_ml_bg(img, zero_median=False)
        assert not np.allclose(result_on, result_off)

    def test_zero_median_true_centres_3d_slices(self, mock_load_unet):
        """Test zero_median=True centres each slice independently in a 3D stack."""
        stack = np.random.rand(3, 256, 256).astype(np.float32) + 10.0
        result = level_ml_bg(stack, zero_median=True)
        for i in range(result.shape[0]):
            assert abs(np.median(result[i])) < abs(np.median(stack[i]))

    def test_zero_median_false_does_not_centre_3d_output(self, mock_load_unet):
        """Test zero_median=False doesn't subtract a per-slice median in a 3D stack."""
        stack = np.random.rand(3, 256, 256).astype(np.float32)
        result_on = level_ml_bg(stack, zero_median=True)
        result_off = level_ml_bg(stack, zero_median=False)
        assert not np.allclose(result_on, result_off)

    def test_zero_median_does_not_affect_background_output(self, mock_load_unet):
        """zero_median flag should have no effect when background=True."""
        img = np.random.rand(256, 256).astype(np.float32)
        bg_on = level_ml_bg(img, background=True, zero_median=True)
        bg_off = level_ml_bg(img, background=True, zero_median=False)
        assert np.allclose(bg_on, bg_off)
