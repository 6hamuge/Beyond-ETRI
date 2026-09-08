"""Sensor-specific sleep proxies: screen, ambient light, heart rate, Wi-Fi, app usage.

Time-of-day bands used throughout: evening 20-23h, late-evening 22-23h,
night 00-05h, morning 05-10h. Thresholds (lux 10 for "dark", HR 35th
percentile for "sleep HR", etc.) are heuristic and qualitatively tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ensure_datetime, to_list


def extract_screen_sleep_features(
    df: pd.DataFrame, subject_col: str = "subject_id"
) -> pd.DataFrame:
    """Screen on/off usage around bedtime and wake (WASO / bedtime proxies)."""
    df = ensure_datetime(df)
    rows = []

    for (subj, date), g in df.groupby([subject_col, "date"]):
        g = g.sort_values("timestamp").copy()

        screen = pd.to_numeric(g["m_screen_use"], errors="coerce").fillna(0)
        hours = g["hour"].values
        times = g["timestamp"].values
        is_on = screen > 0

        evening_mask = (hours >= 20) & (hours <= 23)
        late_evening_mask = (hours >= 22) & (hours <= 23)
        night_mask = (hours >= 0) & (hours <= 5)
        morning_mask = (hours >= 5) & (hours <= 10)

        on_times = pd.to_datetime(times[is_on.values])
        if len(on_times) > 0:
            last_on = max(on_times)
            last_on_hour = last_on.hour + last_on.minute / 60
            if last_on_hour < 6:
                last_on_hour += 24
        else:
            last_on_hour = 0

        morning_on_times = pd.to_datetime(times[(is_on.values) & morning_mask])
        if len(morning_on_times) > 0:
            first_morning_on = min(morning_on_times)
            first_morning_on_hour = (
                first_morning_on.hour + first_morning_on.minute / 60
            )
        else:
            first_morning_on_hour = 0

        rows.append(
            {
                "subject_id": subj,
                "date": date,
                "screen_sleep__on_count_total": int(is_on.sum()),
                "screen_sleep__on_ratio_total": float(is_on.mean()) if len(is_on) else 0,
                "screen_sleep__on_count_20_23": int(is_on[evening_mask].sum())
                if evening_mask.sum()
                else 0,
                "screen_sleep__on_ratio_20_23": float(is_on[evening_mask].mean())
                if evening_mask.sum()
                else 0,
                "screen_sleep__on_count_22_23": int(is_on[late_evening_mask].sum())
                if late_evening_mask.sum()
                else 0,
                "screen_sleep__on_ratio_22_23": float(is_on[late_evening_mask].mean())
                if late_evening_mask.sum()
                else 0,
                "screen_sleep__on_count_00_05": int(is_on[night_mask].sum())
                if night_mask.sum()
                else 0,
                "screen_sleep__on_ratio_00_05": float(is_on[night_mask].mean())
                if night_mask.sum()
                else 0,
                "screen_sleep__last_on_hour": last_on_hour,
                "screen_sleep__first_morning_on_hour": first_morning_on_hour,
            }
        )

    result = pd.DataFrame(rows).fillna(0)
    print(f"[screen_sleep]: {result.shape[1] - 2} features")
    return result


def extract_light_sleep_features(
    df: pd.DataFrame, value_col: str = "m_light", subject_col: str = "subject_id"
) -> pd.DataFrame:
    """Ambient-light brightness / darkness ratios around bedtime."""
    df = ensure_datetime(df)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
    df[value_col] = df[value_col].clip(lower=0, upper=10000)

    rows = []
    for (subj, date), g in df.groupby([subject_col, "date"]):
        g = g.sort_values("timestamp").copy()
        light = g[value_col].values
        hours = g["hour"].values
        times = g["timestamp"].values

        evening_mask = (hours >= 20) & (hours <= 23)
        late_evening_mask = (hours >= 22) & (hours <= 23)
        night_mask = (hours >= 0) & (hours <= 5)
        dark_mask = light <= 10

        bright_times = pd.to_datetime(times[light > 10])
        if len(bright_times) > 0:
            last_bright = max(bright_times)
            last_bright_hour = last_bright.hour + last_bright.minute / 60
            if last_bright_hour < 6:
                last_bright_hour += 24
        else:
            last_bright_hour = 0

        rows.append(
            {
                "subject_id": subj,
                "date": date,
                "light_sleep__mean_total": np.mean(light) if len(light) else 0,
                "light_sleep__max_total": np.max(light) if len(light) else 0,
                "light_sleep__std_total": np.std(light) if len(light) else 0,
                "light_sleep__mean_20_23": np.mean(light[evening_mask])
                if evening_mask.sum()
                else 0,
                "light_sleep__max_20_23": np.max(light[evening_mask])
                if evening_mask.sum()
                else 0,
                "light_sleep__mean_22_23": np.mean(light[late_evening_mask])
                if late_evening_mask.sum()
                else 0,
                "light_sleep__max_22_23": np.max(light[late_evening_mask])
                if late_evening_mask.sum()
                else 0,
                "light_sleep__mean_00_05": np.mean(light[night_mask])
                if night_mask.sum()
                else 0,
                "light_sleep__max_00_05": np.max(light[night_mask])
                if night_mask.sum()
                else 0,
                "light_sleep__dark_ratio_total": np.mean(dark_mask)
                if len(dark_mask)
                else 0,
                "light_sleep__dark_ratio_22_23": np.mean(dark_mask[late_evening_mask])
                if late_evening_mask.sum()
                else 0,
                "light_sleep__dark_ratio_00_05": np.mean(dark_mask[night_mask])
                if night_mask.sum()
                else 0,
                "light_sleep__last_bright_hour": last_bright_hour,
            }
        )

    result = pd.DataFrame(rows).fillna(0)
    print(f"[light_sleep]: {result.shape[1] - 2} features")
    return result


def estimate_sleep_from_hr(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    min_sleep_duration_min: int = 60,
    low_hr_window_min: int = 15,
) -> pd.DataFrame:
    """Estimate the main sleep interval from a sustained drop in heart rate.

    The longest run of 1-minute HR samples below the subject's 35th-percentile
    HR (within a prev-day-20h .. day-12h window) is taken as the sleep period;
    onset/offset/duration and HR-shape features are derived from it.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.explode("heart_rate")
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df = df.dropna(subset=["heart_rate", "timestamp"])
    df = df[df["heart_rate"].between(30, 220)]
    df = df.sort_values([subject_col, "timestamp"])

    results = []
    for subj, subj_df in df.groupby(subject_col):
        sleep_threshold = subj_df["heart_rate"].quantile(0.35)
        dates = subj_df["timestamp"].dt.date.unique()

        for date in dates:
            date = pd.Timestamp(date)
            window_start = date - pd.Timedelta(hours=4)
            window_end = date + pd.Timedelta(hours=12)

            window = subj_df[
                (subj_df["timestamp"] >= window_start)
                & (subj_df["timestamp"] <= window_end)
            ].copy()
            if len(window) < 10:
                continue

            window = window.set_index("timestamp")["heart_rate"]
            window_1min = window.resample("1min").mean().interpolate()
            is_low = window_1min < sleep_threshold

            sleep_onset = sleep_offset = None
            max_duration = 0
            in_sleep = False
            seg_start = None
            for t, low in is_low.items():
                if low and not in_sleep:
                    in_sleep = True
                    seg_start = t
                elif not low and in_sleep:
                    duration = (t - seg_start).total_seconds() / 60
                    if duration > max_duration and duration >= min_sleep_duration_min:
                        max_duration = duration
                        sleep_onset = seg_start
                        sleep_offset = t
                    in_sleep = False

            row = {subject_col: subj, "date": date}
            if sleep_onset and sleep_offset:
                sleep_hr = window_1min[sleep_onset:sleep_offset]
                pre_sleep_hr = window_1min[
                    max(window_1min.index[0], sleep_onset - pd.Timedelta(hours=1)):
                    sleep_onset
                ]
                row.update(
                    {
                        "hr__sleep_onset_hour": sleep_onset.hour + sleep_onset.minute / 60,
                        "hr__sleep_offset_hour": sleep_offset.hour
                        + sleep_offset.minute / 60,
                        "hr__est_tst_min": max_duration,
                        "hr__sleep_hr_mean": sleep_hr.mean(),
                        "hr__sleep_hr_std": sleep_hr.std(),
                        "hr__sleep_hr_min": sleep_hr.min(),
                        "hr__presleep_hr_drop": (
                            pre_sleep_hr.mean() - sleep_hr.mean()
                            if len(pre_sleep_hr) > 0
                            else np.nan
                        ),
                        "hr__arousal_count": int((sleep_hr > sleep_threshold).sum()),
                    }
                )
            else:
                for col in [
                    "hr__sleep_onset_hour",
                    "hr__sleep_offset_hour",
                    "hr__est_tst_min",
                    "hr__sleep_hr_mean",
                    "hr__sleep_hr_std",
                    "hr__sleep_hr_min",
                    "hr__presleep_hr_drop",
                    "hr__arousal_count",
                ]:
                    row[col] = np.nan
            results.append(row)

    result_df = pd.DataFrame(results)
    print(f"[hr_sleep]: {result_df.shape[1] - 2} features, {len(result_df)} rows")
    return result_df


def extract_wifi_bedtime_proxy(
    df: pd.DataFrame, subject_col: str = "subject_id"
) -> pd.DataFrame:
    """Bedtime proxy from night-time Wi-Fi scan gaps.

    NOTE: implemented for completeness but NOT used in the final pipeline
    (see docs/MODEL_DESCRIPTION.md). ``build_feature_matrix`` does not call it.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"].dt.date)
    df["hour"] = df["timestamp"].dt.hour

    results = []
    for (subj, date), grp in df.groupby([subject_col, "date"]):
        night = grp[grp["hour"].between(20, 23) | grp["hour"].between(0, 6)]
        night = night.sort_values("timestamp")
        row = {subject_col: subj, "date": date}

        if len(night) >= 2:
            night = night.set_index("timestamp")
            gaps = night.index.to_series().diff().dt.total_seconds() / 60
            long_gap = gaps[gaps > 30]
            if len(long_gap) > 0:
                gap_start = long_gap.index[0] - pd.Timedelta(
                    minutes=gaps[long_gap.index[0]]
                )
                row["wifi__phone_down_hour"] = gap_start.hour + gap_start.minute / 60
            else:
                row["wifi__phone_down_hour"] = np.nan

            last_scan = night.index[-1]
            row["wifi__last_night_scan_hour"] = last_scan.hour + last_scan.minute / 60

            morning = grp[grp["hour"].between(5, 9)].sort_values("timestamp")
            row["wifi__first_morning_scan_hour"] = (
                morning["timestamp"].iloc[0].hour
                + morning["timestamp"].iloc[0].minute / 60
                if len(morning) > 0
                else np.nan
            )
        else:
            row.update(
                {
                    "wifi__phone_down_hour": np.nan,
                    "wifi__last_night_scan_hour": np.nan,
                    "wifi__first_morning_scan_hour": np.nan,
                }
            )
        results.append(row)

    result_df = pd.DataFrame(results)
    print(f"[wifi_bedtime]: {result_df.shape[1] - 2} features, {len(result_df)} rows")
    return result_df


_SOCIAL_KEYWORDS = [
    "카카오톡", "kakao", "instagram", "인스타",
    "facebook", "messenger", "telegram", "line",
]
_MEDIA_KEYWORDS = [
    "youtube", "유튜브", "netflix", "넷플릭스", "tiktok", "틱톡",
    "wavve", "watcha", "웹툰", "naver", "티빙", "tving",
]


def extract_usage_bedtime_proxy(
    df: pd.DataFrame, subject_col: str = "subject_id"
) -> pd.DataFrame:
    """Bedtime / wake proxies and night app-usage load from usage-stats.

    Last app use -> bedtime; first morning use -> wake; night social/media
    minutes -> sleep-onset-latency / arousal proxies.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"].dt.date)
    df["hour"] = df["timestamp"].dt.hour

    results = []
    for (subj, date), grp in df.groupby([subject_col, "date"]):
        grp = grp.sort_values("timestamp").copy()
        row = {subject_col: subj, "date": date}

        evening = grp[grp["hour"].between(20, 23)].sort_values("timestamp")
        row["usage__last_use_hour"] = (
            evening["timestamp"].iloc[-1].hour
            + evening["timestamp"].iloc[-1].minute / 60
            if len(evening) > 0
            else np.nan
        )
        row["usage__late_evening_count"] = grp["hour"].between(22, 23).sum()
        row["usage__night_count"] = grp["hour"].between(0, 5).sum()
        row["usage__sleep_hour_count"] = grp["hour"].between(0, 6).sum()

        evening_total_time = late_evening_total_time = night_total_time = 0.0
        night_app_count = 0
        night_social_time = night_media_time = 0.0

        for _, r in grp.iterrows():
            hour = r["hour"]
            for app in to_list(r["m_usage_stats"]):
                if not isinstance(app, dict):
                    continue
                app_name = str(app.get("app_name", "")).strip().lower()
                try:
                    total_time = float(app.get("total_time", 0) or 0)
                except (TypeError, ValueError):
                    total_time = 0.0
                if total_time <= 0:
                    continue

                if 20 <= hour <= 23:
                    evening_total_time += total_time
                if 22 <= hour <= 23:
                    late_evening_total_time += total_time
                if 0 <= hour <= 5:
                    night_total_time += total_time
                    night_app_count += 1
                    if any(k in app_name for k in _SOCIAL_KEYWORDS):
                        night_social_time += total_time
                    if any(k in app_name for k in _MEDIA_KEYWORDS):
                        night_media_time += total_time

        row["usage__evening_total_time"] = evening_total_time
        row["usage__late_evening_total_time"] = late_evening_total_time
        row["usage__night_total_time"] = night_total_time
        row["usage__night_app_count"] = night_app_count
        row["usage__night_social_time"] = night_social_time
        row["usage__night_media_time"] = night_media_time

        if len(grp) > 0:
            last_active = grp["timestamp"].iloc[-1]
            last_active_hour = last_active.hour + last_active.minute / 60
            if last_active_hour < 6:
                last_active_hour += 24
            row["usage__last_active_hour"] = last_active_hour
        else:
            row["usage__last_active_hour"] = np.nan

        morning = grp[grp["hour"].between(5, 10)].sort_values("timestamp")
        row["usage__first_morning_hour"] = (
            morning["timestamp"].iloc[0].hour + morning["timestamp"].iloc[0].minute / 60
            if len(morning) > 0
            else np.nan
        )

        if not pd.isna(row["usage__last_use_hour"]) and not pd.isna(
            row["usage__first_morning_hour"]
        ):
            onset = row["usage__last_use_hour"]
            offset = row["usage__first_morning_hour"]
            row["usage__phone_off_duration_hr"] = (
                offset - onset if offset > onset else 24 - onset + offset
            )
        else:
            row["usage__phone_off_duration_hr"] = np.nan

        # total_time is in milliseconds -> hours
        row["usage__evening_total_hr"] = evening_total_time / 3_600_000
        row["usage__late_evening_total_hr"] = late_evening_total_time / 3_600_000
        row["usage__night_total_hr"] = night_total_time / 3_600_000
        row["usage__night_social_hr"] = night_social_time / 3_600_000
        row["usage__night_media_hr"] = night_media_time / 3_600_000

        results.append(row)

    result_df = pd.DataFrame(results).fillna(0)
    print(f"[usage_bedtime]: {result_df.shape[1] - 2} features, {len(result_df)} rows")
    return result_df
