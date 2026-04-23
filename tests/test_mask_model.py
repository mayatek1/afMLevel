"""Tests for ML-based mask levelling via level_ml_mask."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from afmlevel.mask_model import (
    _perimeter_remove,
    _predict_mask_256,
    _process_single_image_mask,
    level_ml_mask,
    ml_edges,
    ml_mask,
)


@pytest.fixture
def mock_load_unet(tiny_unet_cpu):
    """Mock UNet model loading to avoid disk access during tests."""
    with patch(
        "afmlevel.mask_model.load_unet_model",
        return_value=tiny_unet_cpu,
    ) as m:
        yield m


def test_predict_mask_256_with_tiny_unet_output_properties(
    tiny_unet_cpu,
):
    """Tiny U-Net inference returns a 256x256 binary uint8 mask."""
    device = torch.device("cpu")

    image = np.random.rand(256, 256).astype(np.float32)

    mask = _predict_mask_256(
        model=tiny_unet_cpu,
        image_256_norm=image,
        device=device,
        threshold=0.5,
    )

    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert np.all(np.isin(mask, (0, 1)))


def test_predict_mask_256_rejects_wrong_shape(tiny_unet_cpu):
    """Reject inputs that are not exactly 256x256."""
    device = torch.device("cpu")
    bad_image = np.zeros((128, 256), dtype=np.float32)

    with pytest.raises(AssertionError):
        _predict_mask_256(
            model=tiny_unet_cpu,
            image_256_norm=bad_image,
            device=device,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (128, 256),
    ],
)
def test_process_single_image_mask_returns_2d_binary(tiny_unet_cpu, shape):
    """_process_single_image_mask returns a 2D binary uint8 array for valid sizes."""
    img = np.random.rand(*shape).astype(np.float32)

    bin_mask = _process_single_image_mask(tiny_unet_cpu, img)

    assert bin_mask.ndim == 2
    assert bin_mask.shape == shape
    assert bin_mask.dtype == np.uint8
    assert np.all(np.isin(bin_mask, (0, 1)))


def test_2d_input_returns_a_2d_uint8_mask(mock_load_unet):
    """Ensure ml_mask returns a 2D mask with  0 and 1 values from 2D input."""
    img = np.random.rand(256, 256).astype(np.float32)
    mask = ml_mask(img)
    assert mask.ndim == 2
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 1))


def test_3d_input_returns_3d_uint8_mask_array(mock_load_unet):
    """Ensure ml_mask returns a 3D mask array with 0 and 1 values from 3D input."""
    img = np.random.rand(3, 256, 256).astype(np.float32)
    mask = ml_mask(img)
    assert mask.ndim == 3
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 1))


def test_2d_input_returns_a_2d_uint8_mask_edge_array(mock_load_unet):
    """Ensure ml_edges returns a 2D mask with  0 and 1 values from 2D input."""
    img = np.random.rand(256, 256).astype(np.float32)
    edges = ml_edges(img)
    assert edges.ndim == 2
    assert edges.dtype == np.uint8
    assert np.all((edges == 0) | (edges == 1))


def test_3d_input_returns_3d_uint8_mask_edge_array(mock_load_unet):
    """Ensure ml_edges returns a 3D mask array with 0 and 1 values from 3D input."""
    img = np.random.rand(3, 256, 256).astype(np.float32)
    edges = ml_edges(img)
    assert edges.ndim == 3
    assert edges.dtype == np.uint8
    assert np.all((edges == 0) | (edges == 1))


@pytest.mark.parametrize(
    "shape",
    [
        ((2, 3, 4, 5)),
        ((1)),
    ],
)
def test_ml_mask_logs_error_with_invalid_shape(caplog, shape):
    """ml_mask logs an error message when input has invalid dimensionality."""
    bad = np.zeros(shape)

    caplog.set_level("ERROR")

    with pytest.raises(ValueError, match="imarray must be 2D or 3D"):
        ml_mask(bad)

    assert f"Invalid imarray rank for ml_mask: shape={bad.shape}" in caplog.text
    assert str(bad.shape) in caplog.text


def test_ml_mask_casts_to_float32_before_processing(tiny_unet_cpu):
    """Ensure ml_mask passes float32 arrays to _process_single_image_mask."""
    img = np.random.rand(256, 256).astype(np.float64)

    with (
        patch(
            "afmlevel.mask_model.load_unet_model",
            return_value=tiny_unet_cpu,
        ),
        patch(
            "afmlevel.mask_model._process_single_image_mask",
            return_value=np.ones((256, 256), dtype=np.uint8),
        ) as mock_process,
    ):
        ml_mask(img)

        # Ensure _process_single_image_mask was called
        assert mock_process.call_count == 1

        # Extract the image argument passed to _process_single_image_mask
        _, passed_image = mock_process.call_args.args[:2]

        assert passed_image.dtype == np.float32


def test_perimeter_remove_solid_square():
    """Test that a filled square returns only its perimeter pixels."""
    bw = np.zeros((5, 5), dtype=np.uint8)
    bw[1:4, 1:4] = 1

    result = _perimeter_remove(bw)

    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(result, expected)


class TestLevelMlmask:
    """Test suite for the level_ml_mask function."""

    def test_2d_input_returns_2d(self, mock_load_unet):
        """Ensure 2D input returns a 2D output of the same shape."""
        img = np.random.rand(256, 256).astype(np.float32)
        result = level_ml_mask(img)
        assert result.ndim == 2
        assert result.shape == (256, 256)

    def test_3d_input_returns_3d(self, mock_load_unet):
        """Ensure 3D input stacks preserve dimensionality and shape."""
        stack = np.random.rand(3, 256, 256).astype(np.float32)
        result = level_ml_mask(stack)
        assert result.ndim == 3
        assert result.shape == (3, 256, 256)

    def test_invalid_ndim_raises(self, mock_load_unet):
        """Confirm invalid input dimensionality raises a ValueError."""
        with pytest.raises(ValueError, match="2D or 3D"):
            level_ml_mask(np.zeros((2, 3, 4, 5)))

    def test_non_square_512x512(self, mock_load_unet):
        """Ensure a 512x512 image is processed without shape change."""
        img = np.random.rand(512, 512).astype(np.float32)
        result = level_ml_mask(img)
        assert result.shape == (512, 512)

    def test_general_size(self, mock_load_unet):
        """Ensure arbitrary image sizes are preserved after processing."""
        img = np.random.rand(300, 400).astype(np.float32)
        result = level_ml_mask(img)
        assert result.shape == (300, 400)
