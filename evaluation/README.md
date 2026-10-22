# Levelling evaluation (afMLevel paper)

The scripts in this folder generate the quantitative comparison of levelling routines (Figure 4 and SI Table S2) and
the inference timing benchmark (Figure 5B and C) in the afMLevel paper:

> Tekchandani, M., Rollins, D. E., Westerell, M., Evans, S. D., Adams, P. G. & Heath, G. R. afMLevel: Deep learning
> U-Net Models for Levelling Atomic Force Microscopy Images and Movies. bioRxiv (2026).
> <https://doi.org/10.64898/2026.10.19.760138>

They are provided for transparency and are not part of the `afmlevel` Python package. To level your own data, use the
package (`pip install afmlevel`).

## Levelling comparison (MATLAB)

### What the scripts do

Each image in the [164-image test dataset](https://huggingface.co/datasets/Heath-AFM-Lab/afMLevel-data/tree/main/test)
is levelled with 21 routines:

- 14 classical auto-levelling routines from NanoLocz (no mask, threshold mask and multi-plane)
- 4 MLMask routines, which use the mask model to exclude features from plane and line fits
- 3 MLBackground routines, which subtract the background predicted by the background model, applied 1, 2 or 3 times

Each levelled image and its manually levelled ground truth are z-score normalised (mean 0, standard deviation 1). The
mean squared error (MSE) between them is recorded. The result is a table with one row per test image and one column
per routine.

| File | Purpose |
| --- | --- |
| `apply_levelling_and_calculate_MSE.m` | Main script: runs every routine on every test image and writes the MSE table |
| `level_auto_r.m` | The 21 levelling routines, selected by name |
| `thresholder_r.m` | Masking methods; `ML`, `ML edges` and `MLbg` call the models from MATLAB via Python |
| `level.m`, `level_weighted.m` | Plane, line and multi-plane (area-weighted) fitting functions from NanoLocz |
| `afmlevel_functions.py` | Python model inference: `applymodel_mask` (MLMask) and `applymodel_bg` (MLBackground) |
| `lutafm.mat` | AFM colour map for the per-image figures (optional; greyscale is used if it is missing) |
| `unet_timing.py` | Inference timing benchmark through playNano (see [Inference timing](#inference-timing-unet_timingpy)) |
| `GPL-3.0-LICENSE` | GNU GPL v3.0 licence text for the NanoLocz-derived files |

### Requirements

- MATLAB R2021a or later, with the Image Processing Toolbox and Curve Fitting Toolbox
- A Python environment compatible with your MATLAB version
  ([compatibility table](https://www.mathworks.com/support/requirements/python-compatibility.html)) with `numpy`,
  `scipy`, `torch` and `opencv-python` installed
- The trained models from Hugging Face:
  - `mask_unet.pth` from [Heath-AFM-Lab/afMLevel-mask-unet](https://huggingface.co/Heath-AFM-Lab/afMLevel-mask-unet)
  - `background_unet.pth` from
    [Heath-AFM-Lab/afMLevel-background-unet](https://huggingface.co/Heath-AFM-Lab/afMLevel-background-unet)
- The test set from the [afMLevel dataset](https://huggingface.co/datasets/Heath-AFM-Lab/afMLevel-data)

### Setting up

1. **Models.** Put `mask_unet.pth` and `background_unet.pth` in this folder, next to `afmlevel_functions.py`. The
   models are loaded from MATLAB's current folder, and the script changes to this folder before running.
2. **Test data.** Copy the files from the `Image` and `Levelled` folders in the dataset's `test` directory into a single
   folder called `All Test Data`. Each raw image `<image_id>.tiff` needs its ground truth `<image_id>_levelled.tiff` in
   the same folder.
3. **Paths.** At the top of `apply_levelling_and_calculate_MSE.m`, set:
   - `python_path`: the Python executable of your environment (e.g. `...\envs\afmlevel\python.exe`)
   - `script_dir`: this folder
   - `base_path`: the folder that contains `All Test Data`
   - `output_folder`: where the results should be saved (created if it does not exist)

### Running

Run `apply_levelling_and_calculate_MSE.m` in MATLAB. It prints the number of test images found (164 for the full test
set), then processes each image in turn. The MLMask and MLBackground routines call the models for every image, so a
full run takes some time.

### Output

`output_folder` will contain:

- `MSE_results_TEST.csv`: one row per test image (`Image_ID`) and one column per routine, in the order of Figure 4,
  using the routine names from the paper
- `results_<image_id>.png`: for each test image, the raw image, the ground truth and the output of every routine with
  its MSE

From the CSV, the statistics in Figure 4A/B are, for each routine, the mean and median MSE, the standard error of the
mean and the percentage of images with MSE < 0.1. The significance levels in Figure 4C are from Student's t-tests
between routines. The visual inspection scores (Figure 4B) and the sample-type breakdown (Figure 4D) are not produced by
these scripts.

### Routine names

`level_auto_r.m` uses its own names for the routines. The CSV columns use the names from the paper.

| Paper / CSV column | `level_auto_r` routine |
| --- | --- |
| Plane-line | `plane-line` |
| Iterative stdev | `iterative stdev` |
| Iterative fixed high/low | `Iterative fixed high low` |
| Iterative fixed low | `Iterative fixed low` |
| Iterative fixed high | `iterative fixed high` |
| Gaussian high/low | `gaussian high low` |
| Iterative Gaussian low | `iterative gaussian low` |
| Iterative Gaussian high | `iterative gaussian high` |
| Otsu + line | `otsu + line` |
| Iterative Otsu | `iterative otsu` |
| Multi-plane Otsu | `multi-plane otsu` |
| Multi-plane Otsu + line | `multi-plane otsu + line` |
| Multi-plane edges | `multi-plane edges` |
| Multi-plane edges + line | `multi-plane edges + line` |
| Plane-line MLMask | `plane-line MLMask` |
| Iterative MLMask | `iterative MLMask` |
| Multi-plane MLMask | `multi-plane MLMask` |
| Multi-plane MLMask + line | `multi-plane MLMask + line` |
| MLBackground 1x | `MLBackground 1x` |
| MLBackground 2x | `MLBackground 2x` |
| MLBackground 3x | `MLBackground 3x` |

## Inference timing (`unet_timing.py`)

`unet_timing.py` measures how long the afMLevel models take to level high-speed AFM (HS-AFM) videos, using the afMLevel
plugin for [playNano](https://github.com/derollins/playNano). It produced the raw timing data for Figure 5B and C. It was
first shared as a [GitHub Gist](https://gist.github.com/derollins/9c5ec2dfd15dfe209a4733707c9915a8), which is the
version cited in the paper.

### What it does

Each video is loaded with playNano, and the trace and retrace height channels are levelled separately with six
routines:

| Script routine | Paper routine | Steps |
| --- | --- | --- |
| `background_unet_1` | MLBackground 1x | MLBackground model once (3rd-order line fit) |
| `background_unet_2` | MLBackground 2x | MLBackground model twice |
| `background_unet_3` | MLBackground 3x | MLBackground model three times |
| `mask_unet_single` | MLMask | MLMask model, `ML mask` method |
| `mask_it_unet` | Iterative MLMask | MLMask model, `iterative ML mask` method |
| `pnanolocz_auto_itr` | Classical comparison | `iterative 1nm high` routine from pnanolocz (CPU) |

Each routine is run three times per channel (`REPEATS`), and each run is timed. Only the playNano processing pipeline
is timed, not file loading. Before its first timed run, each GPU routine is run once untimed, so that start-up costs
(CUDA initialisation, loading the model) are not included. The GPU is synchronised before and after each timed run.

### Timing requirements

- An NVIDIA GPU and a CUDA-enabled build of PyTorch. The ML routines are set to run on the `cuda` device and fail
  without it.
- `playnano` and `afmlevel` (which installs `pnanolocz`). The paper's timings used playnano 0.3.1 and afmlevel 0.1.1.

```bash
pip install playnano==0.3.1 afmlevel==0.1.1
pip install torch --index-url https://download.pytorch.org/whl/cu128   # choose the build for your CUDA version
python -c "import torch; print(torch.cuda.is_available())"             # should print True
```

The paper's timings were measured on a laptop with an Intel Core Ultra 7 165H processor, 32 GB RAM and an NVIDIA RTX
500-series laptop GPU, running Windows 11. Absolute times will differ on other hardware.

### Timing input data

The script takes a folder containing any of:

- `.h5-jpk` files (JPK/Bruker)
- `.aris` files
- subfolders of `.jpk` frames, one subfolder per video (or pass a single folder of `.jpk` frames directly)

The paper used the public HS-AFM datasets listed in its Methods. Files of other types are skipped.

### Running the benchmark

Before timing, run each model once (for example by running the script on a small video), so that afMLevel has
downloaded the model weights from Hugging Face. Otherwise the first run includes the download. Then run:

```bash
python unet_timing.py "path/to/timing_dataset"
```

Progress and each run's time are printed to the console.

### Timing output

The results are written to `timing_all_data.csv` in the current working directory. Each row is one repeat for one
channel of one video, with columns `filename`, `channel`, `repeat`, `width_px`, `height_px`, `frames` and the time in
seconds for each routine (`<routine>_s`). A routine that fails leaves its cell empty; after a CUDA error, the remaining
GPU routines for that channel are skipped.

The script appends to `timing_all_data.csv` if it already exists, so delete or rename the file before a new run.

For Figure 5, the three repeats were averaged for each video and routine. Figure 5B plots the mean time against the
number of frames, with an ordinary least-squares linear fit for each routine; the slope gives the time per frame.
Figure 5C plots the time per frame against the image size in pixels.

## Licence

`apply_levelling_and_calculate_MSE.m`, `afmlevel_functions.py` and `unet_timing.py` are covered by the repository's BSD
3-Clause licence.

`level_auto_r.m`, `thresholder_r.m`, `level.m` and `level_weighted.m` are derived from
[NanoLocz](https://github.com/George-R-Heath/NanoLocz) (George R. Heath, University of Leeds), which is distributed
under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html). These files are distributed
under the same licence; a copy is included in this folder as `GPL-3.0-LICENSE`.

`thresholder_r.m` and `level_weighted.m` also include code adapted from [SPIW](https://sourceforge.net/projects/spiw/)
(Richard Woolley and Julian Stirling), licensed under the GNU LGPL v3 or later, as stated in their file headers.

## Authors

Maya Tekchandani wrote and ran the levelling comparison scripts. Daniel E. Rollins wrote the inference timing script.
