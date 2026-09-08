"""Synthetic end-to-end smoke test.

Builds small in-memory sensor tables that mimic the ETRI schema and runs the
whole pipeline (feature engineering -> deviation -> dataset -> model -> LOSO-CV
-> ensemble -> calibrated submission). Checks that it runs without error and
that the output is well-formed; it does NOT check score fidelity.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def _synthetic_raw(subjects, days, rng):
    acc: dict[str, list[pd.DataFrame]] = {
        k: []
        for k in [
            "ac_status", "activity", "ambience", "ble", "gps", "light",
            "screen", "usage_stats", "wifi", "hr", "w_light", "pedo",
        ]
    }
    for subj in subjects:
        for day in days:
            n = int(rng.integers(20, 36))
            base = pd.Timestamp(day)
            tss = [base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 1440, n)]
            ms = [int(pd.Timestamp(t).value // 10**6) for t in tss]

            acc["ac_status"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss, "m_charging": rng.integers(0, 2, n)}))
            acc["activity"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss, "m_activity": rng.integers(0, 8, n)}))
            acc["light"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss, "m_light": rng.gamma(2, 50, n)}))
            acc["screen"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss, "m_screen_use": rng.integers(0, 2, n)}))
            acc["w_light"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss, "w_light": rng.gamma(2, 30, n)}))
            acc["pedo"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss, "w_pedo": rng.integers(0, 200, n)}))
            acc["ambience"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss,
                "m_ambience": [[["Speech", float(rng.random())], ["Silence", float(rng.random())]] for _ in range(n)]}))
            acc["ble"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss,
                "m_ble": [[{"address": f"AA:{int(rng.integers(0, 99))}", "rssi": float(-rng.integers(40, 100)),
                            "device_class": int(rng.integers(0, 5))} for _ in range(int(rng.integers(0, 4)))] for _ in range(n)]}))
            acc["gps"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss,
                "m_gps": [[{"speed": float(rng.random() * 5)}] for _ in range(n)]}))
            acc["usage_stats"].append(pd.DataFrame({"subject_id": subj, "timestamp": tss,
                "m_usage_stats": [[{"app_name": random.choice(["kakao", "youtube", "chrome"]),
                                    "total_time": float(rng.integers(0, 600000))}
                                   for _ in range(int(rng.integers(0, 3)))] for _ in range(n)]}))
            acc["wifi"].append(pd.DataFrame({"subject_id": subj, "timestamp": ms, "m_wifi": [[] for _ in range(n)]}))
            acc["hr"].append(pd.DataFrame({"subject_id": subj, "timestamp": ms,
                "heart_rate": [list(rng.integers(50, 90, int(rng.integers(3, 8)))) for _ in range(n)]}))

    return {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["OUT_DIR"] = str(tmp / "out")

    rng = np.random.default_rng(0)
    random.seed(0)
    subjects = [f"S{i:02d}" for i in range(6)]
    days = pd.date_range("2024-05-01", periods=35, freq="D")

    raw = _synthetic_raw(subjects, days, rng)
    labels_df = pd.DataFrame(
        [
            {"subject_id": s, "lifelog_date": d.strftime("%Y-%m-%d"),
             **{c: int(rng.integers(0, 2)) for c in LABELS}}
            for s in subjects for d in days
        ]
    )
    labels_df["date"] = pd.to_datetime(labels_df["lifelog_date"])

    sub = labels_df[labels_df["date"] >= days[-4]][["subject_id", "lifelog_date"]].copy()
    for c in LABELS:
        sub[c] = 0.0
    sub.to_csv(tmp / "ch2026_submission_sample.csv", index=False)

    return raw, labels_df


def test_end_to_end(synthetic):
    raw, labels_df = synthetic

    from src.config import CONFIG, set_seed
    from src.dataset import feature_order
    from src.features import build_feature_matrix
    from src.predict import generate_submission, save_checkpoint, train_seed_ensemble
    from src.train import run_loso_cv, summarize_loso

    CONFIG["epochs"] = 2
    set_seed()

    full_dev_df, raw_cols, dev_cols = build_feature_matrix(raw, labels_df)
    assert len(raw_cols) == len(dev_cols) > 0
    feature_order(raw_cols, dev_cols)

    loso = run_loso_cv(full_dev_df, raw_cols, dev_cols)
    assert loso, "LOSO-CV produced no folds"
    summary = summarize_loso(loso)
    assert "macro" in summary

    models = train_seed_ensemble(full_dev_df, raw_cols, dev_cols, seeds=[42, 7])
    save_checkpoint(models, raw_cols, dev_cols)

    out = generate_submission(models, full_dev_df, raw_cols, dev_cols)
    assert list(out.columns[:2]) == ["subject_id", "lifelog_date"]
    preds = out[LABELS]
    assert preds.notna().all().all()
    # calibration squashes probabilities into [shift, shift + scale] = [0.2, 0.8]
    assert (preds >= 0.2 - 1e-6).all().all() and (preds <= 0.8 + 1e-6).all().all()
