"""Feature engineering pipeline.

``build_feature_matrix`` turns raw sensor tables + labels into the final
model-ready matrix (raw engineered features + personalized-deviation features).
"""

from __future__ import annotations

from functools import reduce

import pandas as pd

from .common import extract_daily_features
from .composite import add_composite_features, add_q1_features, add_rolling_features
from .deviation import merge_labels_and_deviation
from .nested_sensors import (
    extract_ambience_special_features,
    extract_ble_special_features,
    extract_gps_special_features,
)
from .sleep_proxies import (
    estimate_sleep_from_hr,
    extract_light_sleep_features,
    extract_screen_sleep_features,
    extract_usage_bedtime_proxy,
)

__all__ = ["build_daily_features", "build_feature_matrix"]


def build_daily_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge every per-sensor daily feature block into one wide daily matrix.

    Wi-Fi is intentionally excluded (see docs/MODEL_DESCRIPTION.md).
    """
    daily: dict[str, pd.DataFrame] = {}

    # Generic numeric aggregation for every sensor that has numeric columns.
    for key, df in raw.items():
        feat = extract_daily_features(df, sensor_name=key)
        if feat is not None:
            daily[key] = feat

    # Nested-list sensors need dedicated extractors.
    daily["gps_special"] = extract_gps_special_features(raw["gps"])
    daily["ambience_special"] = extract_ambience_special_features(raw["ambience"])
    daily["ble_special"] = extract_ble_special_features(raw["ble"])

    # Sensor-specific sleep proxies.
    daily["screen_sleep"] = extract_screen_sleep_features(raw["screen"])
    daily["light_sleep"] = extract_light_sleep_features(raw["light"])
    daily["usage_bedtime"] = extract_usage_bedtime_proxy(raw["usage_stats"])
    daily["hr_sleep"] = estimate_sleep_from_hr(raw["hr"])

    feature_df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["subject_id", "date"], how="outer"
        ),
        daily.values(),
    )
    print(f"merged daily features: {feature_df.shape}")

    feature_df = add_composite_features(feature_df)
    feature_df = add_rolling_features(feature_df)
    feature_df = add_q1_features(feature_df)
    return feature_df


def build_feature_matrix(
    raw: dict[str, pd.DataFrame], labels_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Full pipeline: raw tables + labels -> ``(full_dev_df, raw_cols, dev_cols)``."""
    feature_df = build_daily_features(raw)
    return merge_labels_and_deviation(feature_df, labels_df)
