"""Sliding-window sequence dataset.

Each sample is the preceding ``seq_len`` days of features (raw block followed
by deviation block) paired with the current day's 7 binary labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import LABEL_COLS


def feature_order(raw_cols: list[str], dev_cols: list[str]) -> list[str]:
    """Model input column order: sorted raw features, then sorted deviation features.

    The model splits its input at the midpoint, so ``len(raw_cols)`` must equal
    ``len(dev_cols)``.
    """
    if len(raw_cols) != len(dev_cols):
        raise ValueError(
            f"raw/dev feature count mismatch: {len(raw_cols)} vs {len(dev_cols)}"
        )
    return [*sorted(raw_cols), *sorted(dev_cols)]


class LifelogDataset(Dataset):
    """Per-subject sliding windows over the deviation-augmented feature matrix."""

    def __init__(
        self,
        df: pd.DataFrame,
        feat_cols: list[str],
        label_cols: list[str] = LABEL_COLS,
        seq_len: int = 7,
        exclude_subjects: list | None = None,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        subjects = df["subject_id"].unique()
        if exclude_subjects:
            subjects = [s for s in subjects if s not in exclude_subjects]

        for subj in subjects:
            subj_df = df[df["subject_id"] == subj].sort_values("date")
            x_arr = subj_df[feat_cols].values.astype(np.float32)

            available = [c for c in label_cols if c in subj_df.columns]
            if available:
                y_arr = subj_df[available].values.astype(np.float32)
            else:
                y_arr = np.zeros((len(subj_df), len(label_cols)), dtype=np.float32)

            x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)
            x_arr = np.clip(x_arr, -10, 10)

            for i in range(seq_len, len(subj_df)):
                y_label = y_arr[i]
                if not np.any(np.isnan(y_label)):
                    self.samples.append(
                        (
                            torch.tensor(x_arr[i - seq_len:i], dtype=torch.float32),
                            torch.tensor(y_label, dtype=torch.float32),
                        )
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
