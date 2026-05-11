# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-11

### Added

First public release

#### Models

- **MLBackground** - U-Net regression model (`afMLevel-background-unet`) that predicts
  tilt, z-scanner drift, and other large-scale imaging artefacts in AFM height maps.
  The predicted background is subtracted to produce a levelled image without altering
  local pixel height values.
- **MLMask** - U-Net segmentation model (`afMLevel-mask-unet`) that generates a binary
  feature mask from AFM height data, enabling masked plane and median line fits as an
  alternative to histogram-based thresholding.
- Pre-trained model weights hosted on [Hugging Face](https://huggingface.co/Heath-AFM-Lab)
  and downloaded automatically on first use via `huggingface_hub`.

#### Processing

- `level_ml_bg()` - apply the background model to a 2D image or 3D HS-AFM stack,
  returning either the levelled image or the predicted background.
- `ml_mask()` - generate a binary background mask (1 = background, 0 = feature) from
  a 2D image or 3D stack using the MLMask model.
- `ml_edges()` - derive an edge mask from the MLMask output using morphological
  operations, for use in multi-plane levelling routines.
- `level_ml_mask()` - apply the MLMask model within a sequence of plane and median
  line fits. Available methods:
  - `iterative-ml-mask` (default)
  - `ml-mask`
  - `multi-plane-ml-mask`
  - `multi-plane-ml-mask-line`
  - `ml-edge-mask`
- Method strings are normalised at runtime (case-insensitive; spaces and underscores
  interchangeable with hyphens).

#### Tiling and preprocessing

- Efficient tiling strategies when using the background model for arbitrary image sizes:
  - Single-tile pass-through for 256 x 256 inputs.
  - Interleaved 4-tile strategy for 512 x 512 inputs, preserving all pixel values
    without seam artefacts.
  - Mirror-pad and grid-tile strategy for general image sizes.
- Nearest-neighbour interpolation was used for resizing image to 256 x 256 for mask model inference.
- Input preprocessing pipeline (plane fit → min-max normalisation → resize/tile)
  applied identically to training and inference data.
- Model instances are cached in-process to avoid repeated loading and Hugging Face
  cache checks when processing stacks.

#### playNano integration

- `afMLevel` registers as a plugin for [playNano](https://github.com/derollins/playNano)
  via the `playnano.video_processing` and `playnano.filters` entry points,
  enabling end-to-end HS-AFM video levelling within the playNano pipeline using
  `level_ml_bg` and `level_ml_mask`.

#### Tests

- Fast unit and integration tests (no model weights required) covering:
  - Tiling round-trips for all tiling modes.
  - Shape preservation for 2D and 3D inputs across all public functions.
  - Mask binarisation, dtype enforcement, and small-object removal.
  - Method string normalisation in `level_ml_mask`.
  - U-Net forward-pass shape and local `.pth` loading.
  - Utility functions (`normalise`, `denormalise`, `xyplanefit`, `swap01`,
    `remove_small_zeros`, `linefit`).
- Slow integration tests (`pytest -m slow`) using real model weights downloaded from
  Hugging Face, including regression sentinels for numerical reproducibility.
- CI: fast tests run on push/PR across Ubuntu, macOS, and Windows for Python 3.11,
  3.12, and 3.13; slow tests run nightly on Ubuntu.

#### Notebooks

- `afMLevel_demo.ipynb` - introduction to the package; applies both models to single
  AFM images, examining all outputs.
- `afMLevel_video_demo.ipynb` - loads HS-AFM videos and applies models via the
  playNano plugin.

#### Documentation

- `README.md` with quick-start guide, dependency table, detailed descriptions of the
  MLBackground and MLMask levelling routines, and contribution guidelines.
- `CITATION.cff` for software citation.
- `CHANGELOG.md` (this file).

[Unreleased]: https://github.com/mayatek1/afMLevel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mayatek1/afMLevel/releases/tag/v0.1.0
