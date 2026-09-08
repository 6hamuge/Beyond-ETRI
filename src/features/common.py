"""Shared utilities and the generic per-sensor daily aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_timestamp_col(df: pd.DataFrame) -> str | None:
    """Best-effort detection of the timestamp column of a sensor table."""
    candidates = ["timestamp", "time", "datetime", "ts", "date"]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def ensure_datetime(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Return a copy with parsed ``timestamp`` plus ``date`` and ``hour`` columns."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["date"] = pd.to_datetime(df[ts_col].dt.date)
    df["hour"] = df[ts_col].dt.hour
    return df


def to_list(x) -> list:
    """Normalize a nested-list / struct sensor cell into a plain ``list``."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


def extract_daily_features(
    df: pd.DataFrame,
    sensor_name: str,
    ts_col: str | None = None,
    subject_col: str = "subject_id",
) -> pd.DataFrame | None:
    """Generic per-subject/day statistics (mean/std/min/max/count) for a sensor.

    Only numeric columns are aggregated; sensors whose payload is a nested
    list/struct (gps, ambience, ble, ...) return ``None`` here and are handled
    by dedicated extractors instead.
    """
    df = df.copy()

    if ts_col is None:
        ts_col = get_timestamp_col(df)

    if ts_col and ts_col in df.columns:
        df[ts_col] = (
            pd.to_datetime(df[ts_col], unit="ms", errors="coerce")
            if df[ts_col].dtype in ["int64", "float64"]
            else pd.to_datetime(df[ts_col], errors="coerce")
        )
        df["date"] = df[ts_col].dt.date
    elif "date" not in df.columns:
        print(f"[{sensor_name}] no timestamp column found.")
        return None

    df["date"] = pd.to_datetime(df["date"])

    exclude = {subject_col, "date", ts_col}
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]
    if not num_cols:
        print(f"[{sensor_name}] no numeric columns.")
        return None

    grp = df.groupby([subject_col, "date"])[num_cols]
    aggs = {
        "mean": grp.mean(),
        "std": grp.std().fillna(0),
        "min": grp.min(),
        "max": grp.max(),
        "count": grp.count(),
    }

    result_parts = []
    for agg_name, agg_df in aggs.items():
        agg_df.columns = [f"{sensor_name}__{c}__{agg_name}" for c in agg_df.columns]
        result_parts.append(agg_df)

    result = pd.concat(result_parts, axis=1).reset_index()
    print(f"[{sensor_name}]: {result.shape[1] - 2} features, {result.shape[0]} rows")
    return result
