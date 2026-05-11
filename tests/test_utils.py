"""Tests for utility functions in the afmlevel.utils module."""

import numpy as np
import pytest

from afmlevel.utils import (
    denormalise,
    linefit,
    normalise,
    remove_small_zeros,
    swap01,
    xyplanefit,
)


def test_normalise_range():
    """Ensure normalise scales data to the range [0, 1]."""
    arr = np.array([[1.0, 3.0], [5.0, 9.0]])
    norm, min_val, data_range = normalise(arr)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)


def test_normalise_constant_array_returns_zeros():
    """Test normalise returns a zero array and data_range=0 for a constant input."""
    arr = np.array([[1.0, 1.0], [1.0, 1.0]])
    norm, min_val, data_range = normalise(arr)
    assert data_range == 0.0
    assert min_val == pytest.approx(1.0)
    np.testing.assert_array_equal(norm, np.zeros_like(arr, dtype=np.float32))


def test_normalise_denormalise_roundtrip():
    """Check that normalise followed by denormalise recovers original data."""
    arr = np.random.rand(64, 64).astype(np.float32) * 100
    norm, min_val, data_range = normalise(arr)
    recovered = denormalise(norm, min_val, data_range)
    np.testing.assert_allclose(recovered, arr, rtol=1e-5)


def test_swap01_swaps_when_more_zeros():
    """Verify swap01 swaps binary values when zeros are in the majority."""
    arr = np.array([[0, 0, 0], [0, 1, 1]], dtype=np.uint8)
    result = swap01(arr)
    assert result[0, 0] == 1


def test_swap01_no_swap_when_more_ones():
    """Verify swap01 leaves data unchanged when ones are in the majority."""
    arr = np.array([[1, 1, 1], [1, 0, 0]], dtype=np.uint8)
    result = swap01(arr)
    assert result[0, 0] == 1


def test_remove_small_zeros_removes_tiny_cluster():
    """Ensure isolated zero pixels below the size threshold are removed."""
    arr = np.ones((10, 10), dtype=np.uint8)
    arr[5, 5] = 0
    result = remove_small_zeros(arr, min_size=5)
    assert result[5, 5] == 1


def test_remove_small_zeros_keeps_large_cluster():
    """Ensure zero regions larger than the minimum size are preserved."""
    arr = np.ones((20, 20), dtype=np.uint8)
    arr[5:10, 5:10] = 0
    result = remove_small_zeros(arr, min_size=10)
    assert result[7, 7] == 0


def test_remove_small_zeros_rejects_non_2d():
    """Confirm remove_small_zeros raises an error for non-2D input."""
    with pytest.raises(ValueError):
        remove_small_zeros(np.ones((3, 3, 3), dtype=np.uint8))


def test_xyplanefit_flat_image_stays_flat():
    """Check that plane fitting leaves a flat image unchanged."""
    arr = np.zeros((32, 32))
    result = xyplanefit(arr, polyx=1, polyy=1)
    np.testing.assert_allclose(result, 0.0, atol=1e-10)


def test_xyplanefit_removes_linear_tilt():
    """Verify xyplanefit removes a linear tilt along the x direction."""
    x = np.arange(32)
    tilt = np.outer(np.ones(32), x * 2.0)
    result = xyplanefit(tilt, polyx=1, polyy=1)
    np.testing.assert_allclose(result, 0.0, atol=1e-8)


def test_linefit_removes_linear_trend_per_row():
    """Test that linefit fits and returns the linear trend for each row."""
    x = np.arange(10, dtype=np.float64)
    arr = np.vstack([x * 2.0, x * 3.0 + 1.0])

    result = linefit(arr, polyx=1)

    assert result.shape == arr.shape
    np.testing.assert_allclose(result, arr, atol=1e-10)


def test_linefit_higher_order():
    """Test that linefit correctly fits a quadratic row."""
    x = np.arange(20, dtype=np.float64)
    row = x**2 - 3 * x + 2.0
    arr = np.vstack([row, row])

    result = linefit(arr, polyx=2)

    np.testing.assert_allclose(result, arr, atol=1e-10)


def test_linefit_output_shape_preserved():
    """Test that linefit returns an array of the same shape as the input."""
    arr = np.random.rand(8, 16)
    result = linefit(arr, polyx=1)
    assert result.shape == arr.shape


@pytest.mark.parametrize("polyx", [0, -1, -10])
def test_linefit_raises_for_invalid_polyx(polyx):
    """Test that linefit raises ValueError for polyx <= 0."""
    arr = np.random.rand(10, 10)
    with pytest.raises(ValueError, match="polyx must be > 0"):
        linefit(arr, polyx=polyx)
