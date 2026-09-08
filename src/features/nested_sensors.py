"""Dedicated extractors for nested-list / struct sensors: GPS, ambience, BLE.

These streams store their payload as lists of dicts (or [label, score] pairs),
so ``select_dtypes`` finds no numeric columns and the generic
``extract_daily_features`` skips them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ensure_datetime, to_list


def extract_gps_special_features(
    df: pd.DataFrame, subject_col: str = "subject_id"
) -> pd.DataFrame:
    """Movement ratio, last-movement hour, and night movement ratio from GPS speed."""
    df = ensure_datetime(df)
    rows = []

    for (subj, date), g in df.groupby([subject_col, "date"]):
        speeds: list[float] = []
        moving_times: list[pd.Timestamp] = []
        night_speed_flags: list[bool] = []

        for arr, ts, hour in zip(
            g["m_gps"].values, g["timestamp"].values, g["hour"].values
        ):
            ts = pd.to_datetime(ts)
            for item in to_list(arr):
                if not isinstance(item, dict) or item.get("speed") is None:
                    continue
                spd = float(item.get("speed"))
                if 0 <= spd <= 30:
                    speeds.append(spd)
                    is_moving = spd > 0.5
                    if is_moving:
                        moving_times.append(ts)
                    if 0 <= hour <= 5:
                        night_speed_flags.append(is_moving)

        speeds = np.array(speeds)
        if len(moving_times) > 0:
            last_move_time = max(moving_times)
            last_movement_hour = last_move_time.hour + last_move_time.minute / 60
            if last_movement_hour < 6:
                last_movement_hour += 24
        else:
            last_movement_hour = 0

        rows.append(
            {
                "subject_id": subj,
                "date": date,
                "gps__moving_ratio": np.mean(speeds > 0.5) if len(speeds) else 0,
                "gps__last_movement_hour": last_movement_hour,
                "gps__night_moving_ratio": np.mean(night_speed_flags)
                if len(night_speed_flags)
                else 0,
            }
        )

    result = pd.DataFrame(rows).fillna(0)
    print(f"[gps_special]: {result.shape[1] - 2} features")
    return result


def extract_ambience_special_features(
    df: pd.DataFrame, subject_col: str = "subject_id"
) -> pd.DataFrame:
    """Acoustic-scene mean scores and top-1 ratios (Speech/Music/Vehicle/Silence/Noise)."""
    df = ensure_datetime(df)
    target_labels = ["Speech", "Music", "Vehicle", "Silence", "Noise"]
    rows = []

    for (subj, date), g in df.groupby([subject_col, "date"]):
        sums = {label: 0.0 for label in target_labels}
        top1_counts = {label: 0 for label in target_labels}
        total_events = 0
        evening_events = 0
        evening_speech = evening_music = evening_vehicle = 0.0
        evening_silence = evening_noise = 0.0

        for arr, hour in zip(g["m_ambience"].values, g["hour"].values):
            scores: dict[str, float] = {}
            for item in to_list(arr):
                item = to_list(item)
                if len(item) >= 2:
                    label = str(item[0])
                    try:
                        score = float(item[1])
                    except (TypeError, ValueError):
                        score = 0.0
                    scores[label] = score
                    if label in target_labels:
                        sums[label] += score

            if scores:
                total_events += 1
                top_label = max(scores, key=scores.get)
                if top_label in top1_counts:
                    top1_counts[top_label] += 1
                if 20 <= hour <= 23:
                    evening_events += 1
                    evening_speech += scores.get("Speech", 0.0)
                    evening_music += scores.get("Music", 0.0)
                    evening_vehicle += scores.get("Vehicle", 0.0)
                    evening_silence += scores.get("Silence", 0.0)
                    evening_noise += scores.get("Noise", 0.0)

        rows.append(
            {
                "subject_id": subj,
                "date": date,
                "ambience__speech__mean_score": sums["Speech"] / total_events
                if total_events
                else 0,
                "ambience__music__mean_score": sums["Music"] / total_events
                if total_events
                else 0,
                "ambience__vehicle__mean_score": sums["Vehicle"] / total_events
                if total_events
                else 0,
                "ambience__silence__mean_score": sums["Silence"] / total_events
                if total_events
                else 0,
                "ambience__noise__mean_score": sums["Noise"] / total_events
                if total_events
                else 0,
                "ambience__speech__top1_ratio": top1_counts["Speech"] / total_events
                if total_events
                else 0,
                "ambience__silence__top1_ratio": top1_counts["Silence"] / total_events
                if total_events
                else 0,
                "ambience__evening_speech_mean": evening_speech / evening_events
                if evening_events
                else 0,
                "ambience__evening_music_mean": evening_music / evening_events
                if evening_events
                else 0,
                "ambience__evening_vehicle_mean": evening_vehicle / evening_events
                if evening_events
                else 0,
                "ambience__evening_silence_mean": evening_silence / evening_events
                if evening_events
                else 0,
                "ambience__evening_noise_mean": evening_noise / evening_events
                if evening_events
                else 0,
            }
        )

    result = pd.DataFrame(rows).fillna(0)
    print(f"[ambience_special]: {result.shape[1] - 2} features")
    return result


def extract_ble_special_features(
    df: pd.DataFrame, subject_col: str = "subject_id"
) -> pd.DataFrame:
    """Nearby-device counts and RSSI statistics from BLE scans (co-presence proxy)."""
    df = ensure_datetime(df)
    rows = []

    for (subj, date), g in df.groupby([subject_col, "date"]):
        scan_device_counts: list[int] = []
        rssis: list[float] = []
        unique_addresses: set = set()
        device_classes: set = set()
        evening_counts: list[int] = []
        night_counts: list[int] = []

        for _, row in g.iterrows():
            hour = row["hour"]
            count = 0
            for dev in to_list(row["m_ble"]):
                if isinstance(dev, dict):
                    count += 1
                    if dev.get("address"):
                        unique_addresses.add(dev.get("address"))
                    if dev.get("device_class") is not None:
                        device_classes.add(str(dev.get("device_class")))
                    if dev.get("rssi") is not None:
                        rssis.append(float(dev.get("rssi")))
            scan_device_counts.append(count)
            if 20 <= hour <= 23:
                evening_counts.append(count)
            if 0 <= hour <= 5:
                night_counts.append(count)

        scan_device_counts = np.array(scan_device_counts)
        rssis = np.array(rssis)

        rows.append(
            {
                "subject_id": subj,
                "date": date,
                "ble__scan_count": len(g),
                "ble__device_count_mean": np.mean(scan_device_counts)
                if len(scan_device_counts)
                else 0,
                "ble__device_count_std": np.std(scan_device_counts)
                if len(scan_device_counts)
                else 0,
                "ble__device_count_max": np.max(scan_device_counts)
                if len(scan_device_counts)
                else 0,
                "ble__unique_device_count": len(unique_addresses),
                "ble__unique_device_class_count": len(device_classes),
                "ble__rssi_mean": np.mean(rssis) if len(rssis) else 0,
                "ble__rssi_std": np.std(rssis) if len(rssis) else 0,
                "ble__rssi_max": np.max(rssis) if len(rssis) else 0,
                "ble__strong_signal_ratio": np.mean(rssis > -60) if len(rssis) else 0,
                "ble__evening_device_count_mean": np.mean(evening_counts)
                if len(evening_counts)
                else 0,
                "ble__night_device_count_mean": np.mean(night_counts)
                if len(night_counts)
                else 0,
            }
        )

    result = pd.DataFrame(rows)
    print(f"[ble_special]: {result.shape[1] - 2} features")
    return result
