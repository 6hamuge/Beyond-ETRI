"""Label merge + personalized-deviation feature construction.

For each retained feature ``x`` of subject ``s`` on day ``t``:

    dev(x, s, t) = clip( (x - mu) / (sigma + eps), -10, 10 )

where ``mu`` / ``sigma`` are a trailing rolling mean / std over the previous
``DEVIATION_WINDOW`` days (``shift(1)`` -> leakage-free), ``eps = 1e-6``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEVIATION_WINDOW, LABEL_COLS, MISSING_RATE_THRESHOLD

_EPS = 1e-6


def merge_labels_and_deviation(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_cols: list[str] = LABEL_COLS,
    window: int = DEVIATION_WINDOW,
    missing_threshold: float = MISSING_RATE_THRESHOLD,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Inner-join labels, drop high-missing features, add ``dev__`` columns.

    Returns ``(full_dev_df, raw_cols, dev_cols)`` where ``raw_cols`` are the
    retained engineered features (sorted) and ``dev_cols`` their deviation
    counterparts (sorted, ``dev__`` prefix).
    """
    label_cols = [c for c in label_cols if c in labels_df.columns]

    full_df = pd.merge(
        feature_df,
        labels_df[["subject_id", "date", *label_cols]],
        on=["subject_id", "date"],
        how="inner",
    )
    print(f"after label merge: {full_df.shape}")

    feat_cols = [c for c in full_df.columns if c not in ["subject_id", "date", *label_cols]]
    for c in feat_cols:
        full_df[c] = full_df[c].replace({True: 1, False: 0})
        full_df[c] = pd.to_numeric(full_df[c], errors="coerce")

    missing_ratio = full_df[feat_cols].isna().mean()
    selected = missing_ratio[missing_ratio < missing_threshold].index.tolist()
    print(f"features before / after missing-rate filter: {len(feat_cols)} / {len(selected)}")

    # Impute: per-subject median -> global median -> 0
    for c in selected:
        full_df[c] = full_df.groupby("subject_id")[c].transform(
            lambda x: x.fillna(x.median())
        )
        full_df[c] = full_df[c].fillna(full_df[c].median()).fillna(0)

    deviation_dfs = []
    for _, subj_df in full_df.groupby("subject_id"):
        subj_df = subj_df.sort_values("date").copy()

        rolling_mean = (
            subj_df[selected].shift(1).rolling(window=window, min_periods=1).mean()
        )
        rolling_std = (
            subj_df[selected]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(_EPS)
        )
        rolling_std = rolling_std.replace(0, _EPS)

        deviation = (subj_df[selected] - rolling_mean) / rolling_std
        deviation = deviation.clip(-10, 10)
        deviation.columns = [f"dev__{c}" for c in selected]

        deviation_dfs.append(
            pd.concat(
                [
                    subj_df[["subject_id", "date", *label_cols]].reset_index(drop=True),
                    subj_df[selected].reset_index(drop=True),
                    deviation.reset_index(drop=True),
                ],
                axis=1,
            )
        )

    full_dev_df = pd.concat(deviation_dfs, ignore_index=True)
    full_dev_df = full_dev_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    raw_cols = sorted(
        c
        for c in full_dev_df.columns
        if c not in ["subject_id", "date", *label_cols] and not c.startswith("dev__")
    )
    dev_cols = sorted(c for c in full_dev_df.columns if c.startswith("dev__"))

    print(f"final matrix: {full_dev_df.shape} | raw={len(raw_cols)} dev={len(dev_cols)}")
    return full_dev_df, raw_cols, dev_cols
