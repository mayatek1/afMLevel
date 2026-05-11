<!-- markdownlint-disable MD033 MD034 -->

# afMLevel: Deep-Learning for Levelling AFM Data

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/afmlevel)](https://pypi.org/project/afmlevel/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/afmlevel)](https://pypi.org/project/afmlevel/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/afmlevel)](https://pypi.org/project/afmlevel/)
[![Tests](https://github.com/mayatek1/afMLevel/actions/workflows/tests.yaml/badge.svg)](https://github.com/mayatek1/afMLevel/actions/workflows/tests.yaml)
[![pre-commit](https://github.com/mayatek1/afMLevel/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/mayatek1/afMLevel/actions/workflows/pre-commit.yaml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Heath--AFM--Lab-yellow)](https://huggingface.co/Heath-AFM-Lab)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

</div>

## **Overview**

AFM image levelling, to correct background tilt and line-by-line scan artifacts, is a critical preprocessing step that
typically requires users to manually apply operations, particularly for complex sample topographies, to ensure accurate
pixel heights. The large frame counts of high-speed AFM movies can make this prohibitively time-consuming. **afMLevel** provides
two deep-learning approaches that fully automate this process without altering local height information, enabling
efficient batch processing of large datasets.

**afMLevel** is a Python package for running two separate trained U-Net models for automatic levelling of AFM images:
**MLMask** and **MLBackground**.

The functions that provide the tools to run these models, including pre-processing and output, are found in the `mask_model`
and `background_model` modules respectively. These tools allow the levelling of AFM images or videos loaded as NumPy arrays.

* `background_model`: Contains the tools for running the **MLBackground** U-Net model on unprocessed AFM image arrays.
  The `level_ml_bg()` function orchestrates the application of this model on raw AFM topography data (2D images and 3D stacks).
* `mask_model`: Includes the `ml_mask()` function and helpers for generating a feature mask from unprocessed AFM image
  arrays with the **MLMask** model. The `level_ml_mask()` uses the output from the `ml_mask()` function within
  auto-levelling routines as an alternative to generating the mask by thresholding (e.g. by Otsu’s method, fixed or
  relative value methods) to process raw AFM images.

![Model overviews](https://raw.githubusercontent.com/mayatek1/afMLevel/main/docs/images/afmlevel_overview.png)

### Model Weights

Pre‑trained PyTorch model weights for the **MLMask** and **MLBackground** U‑Net models, containing the learned network
parameters, are hosted in our [Hugging Face repositories](https://huggingface.co/Heath-AFM-Lab). These are automatically
downloaded and cached when the model is first run and then reused for subsequent applications of the model.

The models are used for inference only; **afMLevel** does not retrain or modify the weights during use.

## **Quick-start guide**

The easiest way to install **afMLevel** is to create a new conda environment and then install the package using pip.

### Create new conda environment

Once you have installed [miniforge](https://conda-forge.org/download/), in the terminal or Miniforge Prompt:

```bash
conda create -n afmlevel-env python=3.11
conda activate afmlevel-env
```

### Installation

The easiest way to install **afMLevel** is from PyPi, simply:

```bash
pip install afmlevel
```

Alternatively you can clone the GitHub repository and install.

Ensure you have [git](https://git-scm.com/) installed on your computer then use the command:

```bash
git clone https://github.com/mayatek1/afMLevel.git
```

Then once the repository is cloned there will be a folder called `afMLevel` containing the **afMLevel** repository.
Navigate into that folder and install with pip.

```bash
cd afMLevel
pip install -e .
```

### Quick usage example

```python
import numpy as np
from afmlevel.background_model import level_ml_bg
from afmlevel.mask_model import level_ml_mask

# imarray is a 2D (H, W) or 3D (N, H, W) NumPy array
levelled = level_ml_bg(imarray)

# Or using the mask-based approach
levelled = level_ml_mask(imarray, method="iterative-ml-mask")
```

### Demonstration Notebooks

A series of Jupyter notebooks have been written to help introduce and demonstrate **afMLevel**. In order to interact with
them you must first install Jupyter notebook in your environment along with packages for opening raw AFM files. This can be
done at installation with the command:

```bash
pip install -e .[notebooks]
```

Then navigate to the notebooks folder within the `afMLevel` repository and launch Jupyter notebook:

```bash
cd afMLevel/notebooks
jupyter notebook
```

Use the browser based interface to open the notebooks and follow the instructions within. Options:

* `afMLevel_demo.ipynb` - An introduction to the project and workflow, loading and applying the ML models to single AFM images,
* `afMLevel_video_demo.ipynb` - Load high-speed AFM videos and apply the ML models through the **afMLevel** plugin for [**playNano**](https://github.com/derollins/playNano)

All functions take a NumPy array as the input (2D - single image; 3D - movie stack) and output the result as a NumPy array.
AFM files can be converted to NumPy arrays via existing software such as [AFMReader](https://github.com/AFM-SPM/AFMReader).
The demo notebooks also give an example of loading AFM files or TIFF files to NumPy arrays.

## Dependencies

**afMLevel** requires Python ≥ 3.11 and the following packages, all installed automatically via pip:

| Package | Purpose |
| --- | --- |
| `torch ≥ 2.0` | U-Net model inference |
| `huggingface_hub` | Automatic model weight download and caching |
| `pnanolocz` | AFM levelling routines used by MLMask methods |
| `numpy`, `scipy` | Array and numerical operations |
| `scikit-image`, `opencv-python-headless` | Image processing utilities |

A GPU is not required — models run on CPU — but inference will be faster with CUDA available.

## MLBackground Levelling Routine

The **MLBackground** model detects noise and imaging artifacts and predicts the image background that contains these
elements. The levelled image is obtained by subtracting third-order polynomial line fits to the predicted background from
the AFM height data.

The model operates on 256 by 256 pixel arrays. To process images of arbitrary size, a combination of reflection padding and
tiling is used to generate multiple input tiles for inference. Details on how pixel values are preserved by splitting alternating
pixels into tiles will be available in the accompanying paper.

The U-Net model [afMLevel-background-unet.pth](https://huggingface.co/Heath-AFM-Lab/afMLevel-background-unet) is then
applied to the 256 by 256 pixel tiles which are then stitched back together. This model detects the noise background
and generates a predicted background for the image. The `level_ml_bg()` function within the `background_model` module
coordinates the generation of this background and subtracts a line fitted version of this from the raw image to give
the levelled image without altering local height differences.

## MLMask Levelling Routine

The **MLMask** model detects image features and produces binary segmentation maps of image features. This is performed by
the `ml_mask()` function, that resizes the input images to 256 by 256 pixels then applies the [afMLevel-mask-unet.pth](https://huggingface.co/Heath-AFM-Lab/afMLevel-mask-unet)
U-Net and generates a binary mask array. That mask is then resized again to the original dimensions. The `ml_edges()`
function uses morphology operations to turn a binary feature mask generated by `ml_mask()` into an edge mask for
region weighted leveling operations.

 The `level_ml_mask()` function coordinates levelling images using various levelling functions from the [pnanolocz](https://github.com/derollins/Python-Nanolocz-Library)
 library, applied with the **MLMask** model generated masks or feature edge masks derived from the model.

The available **afMLevel** mask levelling routines are:

* ml-mask
* iterative-ml-mask (default)
* multi-plane-ml-mask
* multi-plane-ml-mask-line

<!-- markdownlint-disable MD060 -->

| Method                      | Processing steps |
|-----------------------------|------------------|
| ml-mask                     | [1st‑order x–y plane][plane] → single `ml_mask()` → masked [median line subtraction][median] → masked [1st‑order x–y plane][plane] |
| iterative-ml-mask (default) | [1st‑order x–y plane][plane] → 2× (`ml_mask()` + masked [1st‑order x–y plane][plane]) → `ml_mask()` → [median line][median] → masked [1st‑order x plane][plane] → 1× `ml_mask()` → [median line][median] → [2nd‑order x plane][plane] |
| multi-plane-ml-mask         | [1st‑order x–y plane][plane] → 3× (`ml_edges()` + masked [weighted 2nd‑order x–y plane][wplane]) → masked [weighted median line][wmedian] → `ml_edges()` → masked [weighted 2nd‑order x–y plane][wplane] → masked [weighted median line][wmedian] |
| multi-plane-ml-mask-line    | [1st‑order x–y plane][plane] → [median line subtraction][median] → 3× (`ml_edges()` + masked [weighted 2nd‑order x–y plane][wplane]) → masked [weighted median line][wmedian] → `ml_edges()` → masked [weighted 2nd‑order x–y plane][wplane] → masked [weighted median line][wmedian] |

[plane]: https://github.com/derollins/Python-Nanolocz-Library/blob/f3f782ebb32a2c4c80e8135b7f417c25e12f0afc/src/pnanolocz/level.py#L139
[median]: https://github.com/derollins/Python-Nanolocz-Library/blob/f3f782ebb32a2c4c80e8135b7f417c25e12f0afc/src/pnanolocz/level.py#L400
[wplane]: https://github.com/derollins/Python-Nanolocz-Library/blob/f3f782ebb32a2c4c80e8135b7f417c25e12f0afc/src/pnanolocz/level_weighted.py#L247
[wmedian]: https://github.com/derollins/Python-Nanolocz-Library/blob/f3f782ebb32a2c4c80e8135b7f417c25e12f0afc/src/pnanolocz/level_weighted.py#L543

<!-- markdownlint-enable MD060 -->

> **Note:** `level_ml_mask()` uses internal equivalents of `ml_mask()` and `ml_edges()`
> rather than calling them directly, to avoid repeated model loading when processing stacks.
> The public functions remain the recommended entry points for standalone use.

## Use with other software

Since **afMLevel** works primarily with Numpy arrays, in order to load and process raw data from AFM control software
external readers are required to convert the data into an array. This can be done with tools such as
[AFMReader](https://github.com/AFM-SPM/AFMReader), [playNano](https://github.com/derollins/playNano)
and [afmformats](https://github.com/AFM-analysis/afmformats). Examples using **AFMReader** and **playNano** are given in
the [notebooks](#demonstration-notebooks).

**afMLevel** is also supported as a plugin for [playNano](https://github.com/derollins/playNano) out of the box. Simply install
both packages in the same environment and you will be able to use the `level_ml_bg` and `level_ml_mask` functions within
**playNano** straight away. See the [AFMlevel_video_demo notebook](notebooks/afMlevel_video_demo.ipynb) for a programmatic demonstration.

## Contributing and Issues

Bug reports and feature requests are welcome via the
[GitHub Issues page](https://github.com/mayatek1/afMLevel/issues).

To contribute code:

1. Fork the repository and create a branch for your change.
2. Install the development dependencies: `pip install -e ".[dev]"`
3. Run the linters and tests before submitting: `ruff check src/`, `pytest`
4. Open a pull request with a clear description of what the change does and why.

Code style is enforced with `black` and `ruff` (numpy docstring convention).
A `pre-commit` config is included; run `pre-commit install` after cloning to apply checks automatically.

### Tests

Tests are run with pytest and are split into fast (default) and slow categories.
Fast tests use pre-computed reference outputs stored in `tests/` to verify that the
processing pipeline produces correct results without requiring the model weights.
Slow tests run the full inference pipeline by downloading the real model weights from
Hugging Face and are excluded by default to keep CI lightweight and fast.

Regression sentinels in `conftest.py` record expected output values and should only be
updated when the model weights themselves are intentionally updated. A failing sentinel
indicates a change in model behaviour that needs to be investigated.

```bash
pytest                          # fast tests only (slow excluded by default)
pytest -m slow                  # slow tests only
pytest -m "slow or not slow"    # all tests including slow
```

## Citation

If you use **afMLevel** in your research, please cite the software. Citation metadata is provided via the `CITATION.cff`
file in this repository and is automatically recognised by GitHub and reference managers.

A human‑readable citation is provided below for convenience:

Tekchandani, M., Rollins, D. E., & Heath, G. R. (2026). *afMLevel: Deep-Learning for Levelling AFM Data*. University of Leeds.
https://github.com/mayatek1/afMLevel

### BibTeX

```bibtex
@software{afmlevel,
  title = {afMLevel: Deep-Learning for Levelling AFM Data},
  author = {Tekchandani, Maya and Rollins, Daniel E. and Heath, George R.},
  year = {2026},
  url = {https://github.com/mayatek1/afMLevel},
  institution = {University of Leeds},
  license = {BSD-3-Clause}
}
```
