"""Domain-knowledge composite features, rolling bedtime-regularity features, Q1 features.

These run on the merged daily feature matrix (all sensors joined on
``[subject_id, date]``) and assume the upstream extractors have produced the
columns they reference.
"""

from __future__ import annotations

import pandas as pd

# Columns smoothed with 3-/7-day trailing means (only those actually present
# in the matrix are used).
ROLL_COLS = [
    "gps__last_movement_hour",
    "screen_sleep__last_on_hour",
    "screen_sleep__on_ratio_22_23",
    "screen_sleep__on_ratio_00_05",
    "light_sleep__dark_ratio_22_23",
    "light_sleep__dark_ratio_00_05",
    "ambience__silence__top1_ratio",
]


def add_composite_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Late-night phone-use burden and fatigue / stress proxies (Q targets)."""
    feature_df = feature_df.copy()

    # Q common: night-time smartphone-use burden
    feature_df["q__night_phone_burden"] = (
        feature_df["usage__night_total_hr"]
        + feature_df["screen_sleep__on_ratio_00_05"]
        + feature_df["usage__night_app_count"]
    )

    # Q2 fatigue: previous-night activity + poor morning recovery
    feature_df["q2__fatigue_proxy"] = (
        feature_df["usage__late_evening_total_hr"]
        + feature_df["screen_sleep__on_count_00_05"]
        - feature_df["usage__phone_off_duration_hr"]
    )

    # Q3 stress: late use + night use + HR arousal
    feature_df["q3__stress_proxy"] = (
        feature_df["usage__late_evening_count"]
        + feature_df["usage__night_social_hr"]
        + feature_df["hr__arousal_count"]
    )

    return feature_df


def add_rolling_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add 3-/7-day trailing means for the bedtime-regularity columns."""
    feature_df = feature_df.sort_values(["subject_id", "date"]).copy()

    roll_cols = [c for c in ROLL_COLS if c in feature_df.columns]
    for col in roll_cols:
        feature_df[f"{col}_roll3"] = feature_df.groupby("subject_id")[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        feature_df[f"{col}_roll7"] = feature_df.groupby("subject_id")[col].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )

    print("rolling features added")
    return feature_df


def add_q1_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Q1-specific bedtime-regularity and sleep-hygiene composites."""
    feature_df = feature_df.copy()

    feature_df["q1_bedtime_regularity"] = (
        feature_df["screen_sleep__last_on_hour_roll7"]
        - feature_df["screen_sleep__last_on_hour"]
    ).abs()

    feature_df["q1_sleep_hygiene_score"] = (
        feature_df["light_sleep__dark_ratio_00_05"]
        + feature_df["usage__phone_off_duration_hr"]
        - feature_df["screen_sleep__on_ratio_00_05"]
        - feature_df["usage__night_total_hr"]
    )

    return feature_df
