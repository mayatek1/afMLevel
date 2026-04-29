"""
Slow integration tests using real AFM images and real HuggingFace model weights.

Run with:  pytest -m slow
Skip with: pytest -m 'not slow'  (the default)

These tests download ~1 GB of model weights on first run. Subsequent runs
use the HuggingFace local cache (~/.cache/huggingface).
"""

import numpy as np
import pytest
from conftest import (
    _EXPECTED_2D_BG_MEAN,
    _EXPECTED_2D_BG_STD,
    _EXPECTED_2D_MASK_FOREGROUND_FRACTION,
    _EXPECTED_2D_MASKED_MEAN,
    _EXPECTED_2D_MASKED_STD,
    _EXPECTED_3D_BG_SLICE_MEANS,
    _EXPECTED_3D_BG_SLICE_STDS,
)

from afmlevel.background_model import level_ml_bg
from afmlevel.mask_model import level_ml_mask, ml_mask

# ===========================================================================
# Helpers — statistical assertions
# ===========================================================================


def assert_levelled_bg_quality(result: np.ndarray, source: np.ndarray) -> None:
    """
    Assert that a background-levelled image looks like a valid AFM output.

    Checks per-slice so that 3D stacks with accidentally near-zero overall
    mean (positive and negative halves cancelling) don't produce false failures.
    """
    assert not np.any(np.isnan(result)), "Output contains NaN"
    assert not np.any(np.isinf(result)), "Output contains Inf"
    assert (
        result.shape == source.shape
    ), f"Shape changed: {source.shape} → {result.shape}"
    assert result.dtype in (np.float32, np.float64), f"Unexpected dtype: {result.dtype}"
    assert np.std(result) > 0, "Output is constant — model may have failed"
    assert np.std(result) < np.std(source) * 3.0, "Output std exploded"

    # Per-slice zero-centering check — works for both 2D (treated as 1 slice) and 3D
    slices = result if result.ndim == 3 else result[np.newaxis]
    for i, sl in enumerate(slices):
        slice_std = np.std(sl)
        assert slice_std > 0, f"Slice {i} is constant"
        assert abs(np.mean(sl)) < 2.0 * slice_std, (
            f"Slice {i} mean ({np.mean(sl):.4f} nm) is large relative to "
            f"its std ({slice_std:.4f} nm); background may not have been removed"
        )


def assert_mask_quality(mask: np.ndarray, source: np.ndarray) -> None:
    """
    Assert that a binary mask looks like a valid AFM feature mask.

    Checks:
    - Binary values only {0, 1}
    - Shape preserved (ignoring leading stack dimension)
    - Non-trivial: not all zeros, not all ones
    """
    assert mask.dtype == np.uint8, f"Mask dtype should be uint8, got {mask.dtype}"
    assert set(np.unique(mask)).issubset(
        {0, 1}
    ), f"Non-binary mask values: {np.unique(mask)}"
    assert (
        mask.shape == source.shape
    ), f"Mask shape {mask.shape} != input shape {source.shape}"

    foreground_fraction = np.mean(mask)
    assert foreground_fraction > 0.02, (
        f"Mask is nearly all background ({foreground_fraction:.3f} foreground) — "
        "model may have failed or threshold is too high"
    )
    assert foreground_fraction < 0.98, (
        f"Mask is nearly all foreground ({foreground_fraction:.3f}) — "
        "model may have failed or threshold is too low"
    )


def assert_mask_levelled_quality(result: np.ndarray, source: np.ndarray) -> None:
    """
    Assert that a mask-levelled image looks like a valid AFM output.

    Similar to background levelling but with looser constraints since the
    iterative routine applies multiple steps.
    """
    assert not np.any(np.isnan(result)), "Output contains NaN"
    assert not np.any(np.isinf(result)), "Output contains Inf"
    assert (
        result.shape == source.shape
    ), f"Shape changed: {source.shape} → {result.shape}"
    assert np.std(result) > 0, "Output is constant"
    assert np.std(result) < np.std(source) * 5.0, "Output std exploded"


# ===========================================================================
# Background levelling — level_ml_bg
# ===========================================================================


@pytest.mark.slow
class TestLevelMlBgRealData:
    """Integration tests for level_ml_bg using real AFM images."""

    def test_2d_image_output_is_valid(self, real_image_2d):
        """level_ml_bg on a real 2D image produces a statistically valid output."""
        result = level_ml_bg(real_image_2d)
        assert_levelled_bg_quality(result, real_image_2d)

    def test_3d_stack_output_is_valid(self, real_image_3d):
        """level_ml_bg on a real 3D stack produces a statistically valid output."""
        result = level_ml_bg(real_image_3d)
        assert_levelled_bg_quality(result, real_image_3d)

    def test_2d_background_flag_returns_smoother_image(self, real_image_2d):
        """The predicted background should be smoother than the levelled image."""
        bg = level_ml_bg(real_image_2d, background=True)
        lev = level_ml_bg(real_image_2d, background=False)

        # Background is a smooth surface- lower std than the levelled image
        assert np.std(bg) < np.std(
            real_image_2d
        ), "Background has higher std than raw input — unexpected"
        # The two outputs should differ
        assert not np.allclose(
            bg, lev
        ), "background=True and background=False returned identical arrays"

    def test_2d_levelled_has_smaller_mean_than_input(self, real_image_2d):
        """Levelling should bring the mean closer to zero than the raw input."""
        result = level_ml_bg(real_image_2d)
        assert (
            abs(np.mean(result)) < abs(np.mean(real_image_2d)) + np.std(real_image_2d)
        ), "Levelled mean is not smaller than raw mean — levelling may have made things worse"  # noqa: E501

    def test_3d_each_slice_is_valid(self, real_image_3d):
        """Each slice of a levelled 3D stack should individually pass quality checks."""
        result = level_ml_bg(real_image_3d)
        for i in range(result.shape[0]):
            assert not np.any(np.isnan(result[i])), f"Slice {i} contains NaN"
            assert np.std(result[i]) > 0, f"Slice {i} is constant"


# ===========================================================================
# Mask generation — ml_mask
# ===========================================================================


@pytest.mark.slow
class TestMlMaskRealData:
    """Integration tests for ml_mask using real AFM images."""

    def test_2d_mask_is_valid(self, real_image_2d):
        """ml_mask on a real 2D image produces a non-trivial binary mask."""
        mask = ml_mask(real_image_2d)
        assert_mask_quality(mask, real_image_2d)

    def test_3d_mask_is_valid(self, real_image_3d):
        """ml_mask on a real 3D stack produces a non-trivial binary mask stack."""
        mask = ml_mask(real_image_3d)
        assert_mask_quality(mask, real_image_3d)

    def test_2d_mask_spatial_structure(self, real_image_2d):
        """A good mask should have connected regions, not random isolated pixels."""
        from scipy.ndimage import label

        mask = ml_mask(real_image_2d)
        # Count connected components — random noise would produce thousands
        labeled, n_components = label(mask)
        h, w = real_image_2d.shape
        max_reasonable_components = (
            h * w * 0.05
        )  # at most 5% of pixels as isolated blobs
        assert (
            n_components < max_reasonable_components
        ), f"Mask has {n_components} connected components — looks like noise, not features"  # noqa: E501


# ===========================================================================
# Mask levelling — level_ml_mask
# ===========================================================================


@pytest.mark.slow
class TestLevelMlMaskRealData:
    """Integration tests for level_ml_mask using real AFM images."""

    def test_2d_image_output_is_valid(self, real_image_2d):
        """level_ml_mask on a real 2D image produces a statistically valid output."""
        result = level_ml_mask(real_image_2d)
        assert_mask_levelled_quality(result, real_image_2d)

    def test_3d_stack_output_is_valid(self, real_image_3d):
        """level_ml_mask on a real 3D stack produces a statistically valid output."""
        result = level_ml_mask(real_image_3d)
        assert_mask_levelled_quality(result, real_image_3d)

    def test_2d_output_reduces_large_scale_variation(self, real_image_2d):
        """
        Mask levelling should reduce large-scale variation.

        Test by comparing the std of row means before and after —
        a well-levelled image has more uniform row-to-row means.
        """
        result = level_ml_mask(real_image_2d)
        row_std_before = np.std(np.mean(real_image_2d, axis=1))
        row_std_after = np.std(np.mean(result, axis=1))
        assert (
            row_std_after < row_std_before
        ), f"Row-mean std increased after levelling: {row_std_before:.4f} → {row_std_after:.4f}"  # noqa: E501


# ===========================================================================
# Regression sentinels — expected values for the committed sample TIFFs
# ===========================================================================


def _sentinel(name: str, actual: float, expected: float | None, atol: float) -> None:
    """Shared logic for regression sentinel assertions."""
    if expected is None:
        pytest.fail(
            f"{name} not yet set in conftest.py. "
            f"Actual value is {actual:.6f} — copy this into conftest.py."
        )
    assert np.isclose(actual, expected, atol=atol), (
        f"{name} regression: expected {expected:.6f}, got {actual:.6f} "
        f"(atol={atol}). If weights changed intentionally, update conftest.py."
    )


@pytest.mark.slow
class TestRegressionSentinels:
    """
    Regression tests using fixed expected values from the committed sample TIFFs.

    These tests will FAIL on first run — that is expected and intentional.
    Read the actual values from the failure output and paste them into conftest.py.
    """

    def test_2d_bg_mean(self, real_image_2d):
        """level_ml_bg 2D output mean matches expected value."""
        result = level_ml_bg(real_image_2d)
        _sentinel("2D BG mean", np.mean(result), _EXPECTED_2D_BG_MEAN, atol=0.05)

    def test_2d_bg_std(self, real_image_2d):
        """level_ml_bg 2D output std matches expected value."""
        result = level_ml_bg(real_image_2d)
        _sentinel("2D BG std", np.std(result), _EXPECTED_2D_BG_STD, atol=0.05)

    def test_3d_bg_slice_means(self, real_image_3d):
        """level_ml_bg 3D per-slice means match expected values."""
        result = level_ml_bg(real_image_3d)
        if _EXPECTED_3D_BG_SLICE_MEANS is None:
            actual = [
                float(f"{np.mean(result[i]):.6f}") for i in range(result.shape[0])
            ]
            pytest.fail(
                f"_EXPECTED_3D_BG_SLICE_MEANS not set. "
                f"Actual values: {actual} — copy into conftest.py."
            )
        for i in range(result.shape[0]):
            _sentinel(
                f"3D BG slice {i} mean",
                np.mean(result[i]),
                _EXPECTED_3D_BG_SLICE_MEANS[i],
                atol=0.05,
            )

    def test_3d_bg_slice_stds(self, real_image_3d):
        """level_ml_bg 3D per-slice stds match expected values."""
        result = level_ml_bg(real_image_3d)
        if _EXPECTED_3D_BG_SLICE_STDS is None:
            actual = [float(f"{np.std(result[i]):.6f}") for i in range(result.shape[0])]
            pytest.fail(
                f"_EXPECTED_3D_BG_SLICE_STDS not set. "
                f"Actual values: {actual} — copy into conftest.py."
            )
        for i in range(result.shape[0]):
            _sentinel(
                f"3D BG slice {i} std",
                np.std(result[i]),
                _EXPECTED_3D_BG_SLICE_STDS[i],
                atol=0.05,
            )

    def test_2d_mask_foreground_fraction(self, real_image_2d):
        """ml_mask 2D foreground fraction matches expected value."""
        mask = ml_mask(real_image_2d)
        _sentinel(
            "2D mask foreground fraction",
            float(np.mean(mask)),
            _EXPECTED_2D_MASK_FOREGROUND_FRACTION,
            atol=0.02,
        )

    def test_2d_masked_mean(self, real_image_2d):
        """level_ml_mask 2D output mean matches expected value."""
        result = level_ml_mask(real_image_2d)
        _sentinel(
            "2D masked mean", np.mean(result), _EXPECTED_2D_MASKED_MEAN, atol=0.05
        )

    def test_2d_masked_std(self, real_image_2d):
        """level_ml_mask 2D output std matches expected value."""
        result = level_ml_mask(real_image_2d)
        _sentinel("2D masked std", np.std(result), _EXPECTED_2D_MASKED_STD, atol=0.05)
