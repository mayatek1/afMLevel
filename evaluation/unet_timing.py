"""
Script to benchmark afMLevel and pnanolocz processing pipelines.

Features
--------
- Supports .h5-jpk, .aris, and folders of .jpk frames
- Runs trace and retrace channels separately
- Performs a one-time GPU warm-up before timing
- Repeats each routine N times (default: 3)
- Writes results to a CSV file suitable for later averaging

Requirements
------------
This script benchmarks GPU-accelerated afMLevel routines and therefore
requires:

- An NVIDIA GPU
- A CUDA-enabled PyTorch installation
- playnano and afmlevel

The afMLevel routines tested here are configured to run on the CUDA
device and are not expected to run correctly on systems without CUDA
support.

Setup
-----
Create a Python environment and install the required packages:

    pip install playnano afmlevel

Install a CUDA-enabled build of PyTorch. For example:

    pip install torch --index-url https://download.pytorch.org/whl/cu128

Verify CUDA is available:

    python -c "import torch; print(torch.cuda.is_available())"

This should return:

    True

If CUDA is unavailable or PyTorch was installed without CUDA support,
the afMLevel U-Net routines will fail with errors such as:

    Torch not compiled with CUDA enabled

This script was developed and benchmarked using:

    - playnano 0.3.1
    - afmlevel 0.1.1

Example
-------
    python unet_timing.py "path/to/timing_dataset"

Notes
-----
- Timing includes ProcessingPipeline execution only and excludes file loading.
- Used to generate timing data for the afMLevel paper.
- For inference timing ensure the two models are downloaded and cached locally
(run each of the models once before the test run).

Author: Daniel E. Rollins
"""

from pathlib import Path
from playnano.afm_stack import AFMImageStack
from playnano.processing.pipeline import ProcessingPipeline

import torch
import csv
import time
import logging
import sys

# ---------------------- CUDA HYGIENE ----------------------------------

import gc


def reset_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    gc.collect()


WARMED_UP_ROUTINES = set()
# ----------------------------- CONFIG ---------------------------------

REPEATS = 3  # number of timed runs per routine

MASTER_CSV = "timing_all_data.csv"

CHANNEL_MAP = {
    "h5-jpk": {
        "trace": "height_trace",
        "retrace": "height_retrace",
    },
    "aris": {
        "trace": "HeightTrace",
        "retrace": "HeightRetrace",
    },
    "jpk-folder": {
        "trace": "height_trace",
        "retrace": "height_retrace",
    },
}

PROCESSING_ROUTINES = {
    "background_unet_1": [
        (
            "level_ml_bg_single",
            {
                "line_order": 3,
                "device": "cuda",
            },
        ),
    ],
    "background_unet_2": [
        (
            "level_ml_bg_single",
            {
                "line_order": 3,
                "device": "cuda",
            },
        ),
        (
            "level_ml_bg_single",
            {
                "line_order": 3,
                "device": "cuda",
            },
        ),
    ],
    "background_unet_3": [
        (
            "level_ml_bg_single",
            {
                "line_order": 3,
                "device": "cuda",
            },
        ),
        (
            "level_ml_bg_single",
            {
                "line_order": 3,
                "device": "cuda",
            },
        ),
        (
            "level_ml_bg_single",
            {
                "line_order": 3,
                "device": "cuda",
            },
        ),
    ],
    "mask_it_unet": [
        (
            "level_ml_mask_single",
            {
                "device": "cuda",
                "method": "iterative ML mask",
            },
        ),
    ],
    "mask_unet_single": [
        (
            "level_ml_mask_single",
            {
                "device": "cuda",
                "method": "ML mask",
            },
        ),
    ],
    "pnanolocz_auto_itr": [
        (
            "pnanolocz_level_auto",
            {"routine": "iterative 1nm high"},
        )
    ],
}

# -------------------------- LOGGING -----------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# -------------------------- UTILITIES ---------------------------------


def has_jpk_files(p: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() == ".jpk" for f in p.iterdir())


def load_stack_for_channels(input_path: Path):
    stacks = []

    if input_path.is_file() and input_path.suffix.lower() == ".h5-jpk":
        filetype = "h5-jpk"
    elif input_path.is_file() and input_path.suffix.lower() == ".aris":
        filetype = "aris"
    elif input_path.is_dir() and has_jpk_files(input_path):
        filetype = "jpk-folder"
    else:
        return stacks

    for label, channel in CHANNEL_MAP[filetype].items():
        try:
            stack = AFMImageStack.load_data(input_path, channel=channel)
            stacks.append((stack, label))
        except Exception as e:
            logging.warning(f"Failed to load {label} for {input_path}: {e}")

    return stacks


# ---------------------- CORE TIMING LOGIC ------------------------------


def process_stack(stack: AFMImageStack, channel_label: str):
    cuda_broken = False
    # CSV header (unchanged, but now correct for final layout)
    if not Path(MASTER_CSV).exists():
        with open(MASTER_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "filename",
                    "channel",
                    "repeat",
                    "width_px",
                    "height_px",
                    "frames",
                    *[f"{k}_s" for k in PROCESSING_ROUTINES],
                ]
            )

    n, w, h = stack.data.shape[0], stack.data.shape[-2], stack.data.shape[-1]
    filename = stack.file_path

    # ---------------------------------------------------------
    # LOOP OVER REPEATS FIRST
    # ---------------------------------------------------------
    for repeat in range(1, REPEATS + 1):

        timing_row = {k: "" for k in PROCESSING_ROUTINES}

        for routine_name, routine_steps in PROCESSING_ROUTINES.items():
            uses_cuda = any(
                isinstance(kwargs, dict) and kwargs.get("device") == "cuda"
                for _, kwargs in routine_steps
            )

            if cuda_broken and uses_cuda:
                logging.warning(
                    f"{filename} | {channel_label} | skipping {routine_name} "
                    "due to earlier CUDA failure"
                )
                timing_row[routine_name] = ""
                continue

            # ---------------- GLOBAL GPU WARM-UP (ONCE PER ROUTINE) ----------------
            pipeline = None
            if uses_cuda and routine_name not in WARMED_UP_ROUTINES:
                try:
                    if "raw" in stack.processed:
                        stack.restore_raw()

                    pipeline = ProcessingPipeline(stack)
                    for step, kwargs in routine_steps:
                        pipeline.add_filter(step, **kwargs)
                        if uses_cuda:
                            torch.cuda.synchronize()

                    torch.cuda.synchronize()
                    pipeline.run()
                    torch.cuda.synchronize()

                    WARMED_UP_ROUTINES.add(routine_name)

                except Exception as e:
                    logging.error(f"Warm-up failed for {routine_name}: {e}")
                    reset_cuda()
                    continue

                finally:
                    if pipeline is not None:
                        del pipeline
                    if uses_cuda:
                        reset_cuda()

            # ---------------- TIMED RUN ----------------
            pipeline = None
            try:
                if "raw" in stack.processed:
                    stack.restore_raw()

                pipeline = ProcessingPipeline(stack)
                for step, kwargs in routine_steps:
                    pipeline.add_filter(step, **kwargs)
                    if uses_cuda:
                        torch.cuda.synchronize()

                if uses_cuda:
                    torch.cuda.synchronize()

                start = time.perf_counter()
                pipeline.run()
                if uses_cuda:
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start

                timing_row[routine_name] = f"{elapsed:.6f}"

                logging.info(
                    f"{filename} | {channel_label} | {routine_name} | run {repeat} "
                    f"= {elapsed:.4f} s"
                )

            except Exception as e:
                logging.error(
                    f"{filename} | {channel_label} | {routine_name} | run {repeat} FAILED: {e}"
                )
                timing_row[routine_name] = ""

                if uses_cuda:
                    cuda_broken = True
                    reset_cuda()

            finally:
                if pipeline is not None:
                    del pipeline
                if uses_cuda:
                    reset_cuda()

        # ---------------------------------------------------------
        # WRITE ONE ROW PER REPEAT
        # ---------------------------------------------------------
        with open(MASTER_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    filename,
                    channel_label,
                    repeat,
                    w,
                    h,
                    n,
                    *[timing_row[k] for k in PROCESSING_ROUTINES],
                ]
            )


# --------------------- FILE / FOLDER LOGIC -----------------------------


def process_file_or_folder(input_path: Path):
    stacks = load_stack_for_channels(input_path)

    if not stacks:
        logging.info(f"Skipping {input_path}")
        return

    for stack, channel in stacks:
        logging.info(f"Processing {input_path} | channel={channel}")
        process_stack(stack, channel)


def process_folder(folder: Path):
    folder = Path(folder)

    if folder.is_dir() and has_jpk_files(folder):
        process_file_or_folder(folder)
        return

    for f in folder.glob("*.h5-jpk"):
        process_file_or_folder(f)

    for f in folder.glob("*.aris"):
        process_file_or_folder(f)

    for sub in folder.iterdir():
        if sub.is_dir() and has_jpk_files(sub):
            process_file_or_folder(sub)


# ------------------------------ MAIN ----------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch timing AFM pipelines")
    parser.add_argument("folder", type=str, help="Input folder")

    args = parser.parse_args()
    process_folder(Path(args.folder))
