"""Load raw sensor tables and training labels from :data:`config.DATA_DIR`."""

from __future__ import annotations

import os
import warnings

import pandas as pd

from .config import DATA_DIR, FILE_MAP, LABEL_COLS, LABEL_FILE, data_path

warnings.filterwarnings("ignore")


def load_sensor_tables(data_dir: str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read every available Parquet sensor table listed in :data:`FILE_MAP`.

    Missing files are skipped (the pipeline only requires the streams it
    actually consumes), so the returned dict may have fewer than 12 keys.
    """
    raw: dict[str, pd.DataFrame] = {}
    for key, fname in FILE_MAP.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            raw[key] = pd.read_parquet(path)
    print(f"Loaded sensor tables: {len(raw)}/{len(FILE_MAP)}")
    return raw


def load_labels(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Read ``ch2026_metrics_train.csv`` and normalize its date column.

    Adds a ``date`` column (``datetime64``) derived from ``lifelog_date`` so it
    can be merged with the daily feature matrix on ``[subject_id, date]``.
    """
    path = data_path(LABEL_FILE) if data_dir == DATA_DIR else os.path.join(data_dir, LABEL_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    labels_df = pd.read_csv(path)
    labels_df["date"] = pd.to_datetime(labels_df["lifelog_date"])
    keep = ["subject_id", "date", *[c for c in LABEL_COLS if c in labels_df.columns]]
    return labels_df[keep]
