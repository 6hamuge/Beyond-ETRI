"""Train the multi-seed ensemble on all data and write the calibrated submission.

Usage:
    python scripts/make_submission.py       # reads /data, writes /data/out
    DATA_DIR=./data/ch2025 OUT_DIR=./out python scripts/make_submission.py

Reproduces ``PTDT_seq7_ensemble_smooth06.csv`` (competition leaderboard 0.63265).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import set_seed  # noqa: E402
from src.data_loading import load_labels, load_sensor_tables  # noqa: E402
from src.features import build_feature_matrix  # noqa: E402
from src.predict import generate_submission, save_checkpoint, train_seed_ensemble  # noqa: E402


def main() -> None:
    set_seed()
    raw = load_sensor_tables()
    labels_df = load_labels()

    full_dev_df, raw_cols, dev_cols = build_feature_matrix(raw, labels_df)

    models = train_seed_ensemble(full_dev_df, raw_cols, dev_cols)
    save_checkpoint(models, raw_cols, dev_cols)
    generate_submission(models, full_dev_df, raw_cols, dev_cols)


if __name__ == "__main__":
    main()
