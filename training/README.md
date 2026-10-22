# Training the afMLevel models

This folder contains the scripts used to train the two U-Net models distributed with **afMLevel** and described in the
afMLevel paper:

| Script | Model | Loss |
| --- | --- | --- |
| `MaskAire_train34_VaryAll.py` | **MLMask** – binary feature/background segmentation | BCE + Dice |
| `BGAire_train34_VaryAll_2.py` | **MLBackground** – background regression | MSE |

Both models were trained on Aire, the University of Leeds High Performance Computing (HPC) facility, using the
[afMLevel-data](https://doi.org/10.57967/hf/11026) training dataset.

The scripts are as they were on Aire, with the values used in the published models. The only change is the
removal of an exclusion list that referred to specific file names, which are not used in the published
dataset. The list never matched any file, so removing it does not change the behaviour.

## Published models

The published weights are these training checkpoints:

* **MLMask:** `unetmodel_Aire_Mask_80eps_b32_d0_1f7_f7_7l_line2_train34_otsublur1_epoch69.pth`, distributed as
  [`mask_unet.pth`](https://huggingface.co/Heath-AFM-Lab/afMLevel-mask-unet)
* **MLBackground:** `unetmodel_Aire_BG_60eps_b32_d0_1f9_f9_7l_MSE_0line_train34_epoch59.pth`, distributed as
  [`background_unet.pth`](https://huggingface.co/Heath-AFM-Lab/afMLevel-background-unet)

| | MLMask | MLBackground |
| --- | --- | --- |
| U-Net levels | 7 (16 → 1024 channels) | 7 (16 → 1024 channels) |
| Kernel size | 7 × 7 | 9 × 9 |
| Activation | ReLU | ReLU |
| Dropout | 0 | 0 |
| Line fit before training | No | No |
| Line-noise augmentations per image | 2 | 0 |
| Batch size | 32 | 32 |
| Epochs run | 80 | 60 |
| Checkpoint used | Epoch 69 | Epoch 59 |
| Optimiser | Adam, learning rate 0.0005 | Adam, learning rate 0.0005 |

Each script saves a checkpoint for each of the last few epochs (15 for MLMask, 20 for MLBackground). The published
checkpoint was chosen by hand as the one with the lowest validation loss among the final epochs; the scripts save the
candidates but do not make this choice.

The files on Hugging Face are byte-for-byte identical to these training checkpoints. To check that a copy you have is
the published version, compare its SHA-256 checksum (`sha256sum` on Linux/macOS, `Get-FileHash` in PowerShell) with
these values:

* **MLMask** (`mask_unet.pth`, 627,862,402 bytes)
  * Hugging Face revision: `5bdd639c1ea9cd6ee20b0ce67a277ff4e595159e`
  * SHA-256: `27a2c72fafa049025890366be8186d5773a559cf757dd6876c2732a1098f9fdc`
* **MLBackground** (`background_unet.pth`, 1,030,439,618 bytes)
  * Hugging Face revision: `a59724dd87401a7183e0ea4f7da9ec1fa9f5db1f`
  * SHA-256: `094c2ddf4717c393014f24bebeaa51f183b9a6ae59d8a01defcd6e000fade5e8`

## Data

The scripts expect a data directory laid out as:

```text
<data-dir>/
├── Image/       raw AFM images, e.g. name.tif
└── Levelled/    levelled versions, e.g. name_levelled.tif
```

Every file in `Image/` must have a matching `<name>_levelled<ext>` file in `Levelled/`, in the same height units.

With the 2,001 images of the published dataset, the 80:20 split gives 1,600 training and 401 validation images.

## Method

### Preprocessing

Both models use the same input preprocessing:

1. Resize the raw image to 256 × 256 pixels with nearest-neighbour interpolation.
2. Subtract a first-order polynomial fitted to the column means, then one fitted to the row means of the result
   (`xyplanefit`, a first-order plane fit).
3. Min–max normalise to [0, 1].

The levelled image is resized in the same way and used to make the ground truth:

* **MLMask:** the levelled heights are mean-centred, passed through a sigmoid, scaled to 8-bit, blurred with a
  Gaussian (σ = 1 pixel) and thresholded with Otsu's method. Pixels above the threshold are 1, others 0. The sigmoid
  acts on raw height values, so the mask depends on the height units of the levelled images.
* **MLBackground:** the levelled image is scaled with the minimum and maximum of the plane-fitted raw image, so both
  share one scale. The ground truth is the normalised raw image minus the scaled levelled image.

### Split and augmentation

The images are split 80:20 into training and validation sets with scikit-learn's `train_test_split` and
`random_state=42`.

Both subsets are then augmented. Each image contributes, in order: the original, a 180° rotation, the line-noise copies
and a horizontal flip. The line-noise copies have 1–3 horizontal bands, each 5–50 rows tall, shifted up or down by a
random height. For MLMask the copies keep the original mask; for MLBackground the bands are also added to the
background.

| | MLMask | MLBackground |
| --- | --- | --- |
| Copies per image | 5 | 3 |
| Training samples | 8,000 | 4,800 |
| Validation samples | 2,005 | 1,203 |

### Loss functions

* **MLMask:** binary cross-entropy on the network output plus soft Dice loss on its sigmoid. Dice is computed per
  image and averaged over the batch, where p is the predicted probability and y the ground truth:

  ```text
  L_Dice = 1 - (2 Σ p·y + 1) / (Σ p + Σ y + 1)
  ```

* **MLBackground:** mean squared error. The script also supports the BCE + Dice loss and SSIM (via `piqa`); these were
  tried during development but not used for the published model.

Validation losses are computed on the augmented validation set.

## Running the scripts

Install the dependencies:

```bash
pip install torch numpy scipy scikit-image scikit-learn pillow pandas matplotlib piqa
```

The scripts take no command-line options. Run each one from the data directory, the folder containing `Image/` and
`Levelled/`:

```bash
cd /path/to/data
python /path/to/afMLevel/training/MaskAire_train34_VaryAll.py
python /path/to/afMLevel/training/BGAire_train34_VaryAll_2.py
```

Settings are the variables in the `~~DEFINE VARIABLES~~` block near the top of each script, and are set to the published
configuration. The scripts build a run label from these settings and use it in every output file name; for the
published settings it matches the checkpoint names above. A GPU is used if one is available. Outputs are written to the
current folder:

| File | Contents |
| --- | --- |
| `unetmodel_<label>_epoch<N>.pth` | Checkpoints for the last few epochs |
| `unetmodel_<label>.pth` | Final-epoch checkpoint |
| `Lossfinal_<label>.csv`, `.png` | Training and validation loss for every epoch |
| `Validation_<label>_<batch>.png` | Example validation predictions from the final epoch |

The 7-level models are large (about 157 million parameters for MLMask and 258 million for MLBackground), so training
at the published settings needs a GPU with plenty of memory. Set `testrun = True` for a quick check on a small subset.

The checkpoints load directly into `afmlevel.unet.UNet`, which has the same layer names:

```python
from afmlevel.unet import load_unet_model

model = load_unet_model(
    "unetmodel_<label>_epoch69.pth",
    n_channels=1,
    config={"filtersize": 7},
    device="cpu",
)
```

## Reproducibility notes

Retraining will not give the published weights bit for bit:

* No random seed is set for PyTorch or Python's `random` module, so weight initialisation, batch order and the
  line-noise augmentations differ between runs.
* Results also depend on the PyTorch version and GPU.

## Authors

Maya Tekchandani wrote and ran the training scripts. Daniel E. Rollins prepared them for publication.
