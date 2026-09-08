"""Central configuration: paths, file map, labels, hyperparameters, seeding.

Data I/O is anchored at ``/data`` per the competition submission rules.
Override locally with the ``DATA_DIR`` / ``OUT_DIR`` environment variables,
e.g. ``DATA_DIR=./data/ch2025 python scripts/run_loso_cv.py``.
"""

from __future__ import annotations

import os
import random

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR: str = os.environ.get("DATA_DIR", "/data")
OUT_DIR: str = os.environ.get("OUT_DIR", "/data/out")

# Raw sensor tables (Parquet). 9 mobile + 3 wearable streams.
FILE_MAP: dict[str, str] = {
    "ac_status": "ch2025_mACStatus.parquet",
    "activity": "ch2025_mActivity.parquet",
    "ambience": "ch2025_mAmbience.parquet",
    "ble": "ch2025_mBle.parquet",
    "gps": "ch2025_mGps.parquet",
    "light": "ch2025_mLight.parquet",
    "screen": "ch2025_mScreenStatus.parquet",
    "usage_stats": "ch2025_mUsageStats.parquet",
    "wifi": "ch2025_mWifi.parquet",
    "hr": "ch2025_wHr.parquet",
    "w_light": "ch2025_wLight.parquet",
    "pedo": "ch2025_wPedo.parquet",
}

LABEL_FILE: str = "ch2026_metrics_train.csv"
SUBMISSION_SAMPLE_FILE: str = "ch2026_submission_sample.csv"

# --------------------------------------------------------------------------
# Task definition
# --------------------------------------------------------------------------
# Q1-Q3: subjective (sleep quality, fatigue, stress)
# S1-S4: objective sleep sub-indicators
LABEL_COLS: list[str] = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
# Trailing rolling window (days) for the personalized-deviation feature.
DEVIATION_WINDOW: int = 7
# Features with a higher missing rate than this are dropped before deviation.
MISSING_RATE_THRESHOLD: float = 0.7

# --------------------------------------------------------------------------
# Model / training hyperparameters (single unified PTDTransformer)
# --------------------------------------------------------------------------
CONFIG: dict[str, float | int] = {
    "seq_len": 7,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 128,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "batch_size": 16,
    "patience": 7,
}

# Base seed for LOSO-CV; ensemble seeds for the final submission model.
SEED: int = 42
ENSEMBLE_SEEDS: list[int] = [42, 7, 2025, 123, 999]

# Probability calibration applied to the ensembled submission:
#   p_out = p * CALIBRATION_SCALE + CALIBRATION_SHIFT
CALIBRATION_SCALE: float = 0.6
CALIBRATION_SHIFT: float = 0.2

SUBMISSION_NAME: str = "PTDT_seq7_ensemble_smooth06.csv"


def set_seed(seed: int = SEED) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` for reproducible runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_path(name: str) -> str:
    """Absolute path to a file inside :data:`DATA_DIR`."""
    return os.path.join(DATA_DIR, name)
