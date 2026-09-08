"""Run the full pipeline through LOSO-CV and print the results table.

Usage:
    python scripts/run_loso_cv.py            # reads from /data
    DATA_DIR=./data/ch2025 python scripts/run_loso_cv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import set_seed  # noqa: E402
from src.data_loading import load_labels, load_sensor_tables  # noqa: E402
from src.features import build_feature_matrix  # noqa: E402
from src.train import run_loso_cv, summarize_loso  # noqa: E402


def main() -> None:
    set_seed()
    raw = load_sensor_tables()
    labels_df = load_labels()

    full_dev_df, raw_cols, dev_cols = build_feature_matrix(raw, labels_df)
    loso_results = run_loso_cv(full_dev_df, raw_cols, dev_cols)
    summarize_loso(loso_results)


if __name__ == "__main__":
    main()
