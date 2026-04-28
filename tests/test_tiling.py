"""Tests for tiling and stitching helpers in afmlevel.background_model."""

import numpy as np
import pytest

from afmlevel.background_model import _split_into_tiles, _stitch_from_tiles


@pytest.mark.parametrize(
    "shape",
    [
        (256, 256),  # mode=256, single tile
        (512, 512),  # mode=512, interleaved 4-tile
        (300, 400),  # mode=0, general mirror-pad
        (128, 128),  # mode=0, smaller than 256
        (768, 512),  # mode=0, non-square
    ],
)
def test_tiling_roundtrip(shape):
    """Ensure splitting and stitching returns the original image."""
    rng = np.random.default_rng(42)
    image = rng.random(shape).astype(np.float32)
    tiles, meta = _split_into_tiles(image)
    stitched = _stitch_from_tiles(tiles, meta)
    assert stitched.shape == shape
    np.testing.assert_allclose(stitched, image, rtol=1e-5, atol=1e-6)


def test_256_produces_one_tile():
    """Verify a 256x256 image produces a single tile with mode 256."""
    image = np.random.rand(256, 256).astype(np.float32)
    tiles, meta = _split_into_tiles(image)
    assert len(tiles) == 1
    assert meta["mode"] == 256


def test_512_produces_four_tiles():
    """Verify a 512x512 image produces four tiles with mode 512."""
    image = np.random.rand(512, 512).astype(np.float32)
    tiles, meta = _split_into_tiles(image)
    assert len(tiles) == 4
    assert meta["mode"] == 512


def test_all_tiles_are_256x256():
    """Ensure all generated tiles have shape (256, 256)."""
    for shape in [(256, 256), (512, 512), (300, 400)]:
        image = np.random.rand(*shape).astype(np.float32)
        tiles, _ = _split_into_tiles(image)
        for tile in tiles:
            assert tile.shape == (256, 256)
