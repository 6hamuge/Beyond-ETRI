"""Final-model training (multi-seed ensemble) and submission-file generation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    CALIBRATION_SCALE,
    CALIBRATION_SHIFT,
    CONFIG,
    ENSEMBLE_SEEDS,
    LABEL_COLS,
    OUT_DIR,
    SUBMISSION_NAME,
    SUBMISSION_SAMPLE_FILE,
    data_path,
    set_seed,
)
from .dataset import LifelogDataset, feature_order
from .model import PTDTransformer
from .train import get_device, train_one_epoch


def _build_model(n_features: int, n_labels: int, dropout: float, device) -> PTDTransformer:
    return PTDTransformer(
        n_features=n_features,
        n_labels=n_labels,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        dropout=dropout,
        seq_len=CONFIG["seq_len"],
    ).to(device)


def train_seed_ensemble(
    full_dev_df: pd.DataFrame,
    raw_cols: list[str],
    dev_cols: list[str],
    label_cols: list[str] = LABEL_COLS,
    seeds: list[int] = ENSEMBLE_SEEDS,
    device=None,
) -> list[PTDTransformer]:
    """Train one PTDTransformer per seed on the full dataset (no held-out subject)."""
    device = device or get_device()
    feat_cols = feature_order(raw_cols, dev_cols)
    n_features, n_labels = len(feat_cols), len(label_cols)

    models: list[PTDTransformer] = []
    for seed in seeds:
        set_seed(seed)
        ds = LifelogDataset(full_dev_df, feat_cols, label_cols, CONFIG["seq_len"])
        loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True)

        model = _build_model(n_features, n_labels, CONFIG["dropout"], device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
        )
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(CONFIG["epochs"]):
            train_one_epoch(model, loader, optimizer, criterion, device)

        model.eval()
        models.append(model)
        print(f"  seed {seed}: trained ({len(ds)} samples)")
    return models


def save_checkpoint(
    models: list[PTDTransformer],
    raw_cols: list[str],
    dev_cols: list[str],
    label_cols: list[str] = LABEL_COLS,
    path: str | None = None,
) -> str:
    """Persist all ensemble members + config + feature order to a single file."""
    path = path or os.path.join(OUT_DIR, "ptdt_ensemble.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_states": [m.state_dict() for m in models],
            "config": CONFIG,
            "seeds": ENSEMBLE_SEEDS,
            "feat_cols": feature_order(raw_cols, dev_cols),
            "raw_cols": sorted(raw_cols),
            "dev_cols": sorted(dev_cols),
            "label_cols": label_cols,
        },
        path,
    )
    print(f"checkpoint saved: {path}")
    return path


def generate_submission(
    models: list[PTDTransformer],
    full_dev_df: pd.DataFrame,
    raw_cols: list[str],
    dev_cols: list[str],
    label_cols: list[str] = LABEL_COLS,
    out_path: str | None = None,
    device=None,
) -> pd.DataFrame:
    """Predict every row of the submission sample and write the calibrated CSV.

    For each ``(subject, lifelog_date)`` the preceding ``seq_len`` days of
    features are fed to every ensemble member; probabilities are averaged then
    calibrated as ``p * CALIBRATION_SCALE + CALIBRATION_SHIFT``.
    """
    device = device or get_device()
    feat_cols = feature_order(raw_cols, dev_cols)
    seq_len = CONFIG["seq_len"]

    submission = pd.read_csv(data_path(SUBMISSION_SAMPLE_FILE))
    submission["lifelog_date"] = pd.to_datetime(submission["lifelog_date"])

    for m in models:
        m.eval()

    pred_rows = []
    for _, row in submission.iterrows():
        subj = row["subject_id"]
        target_date = row["lifelog_date"]

        hist = full_dev_df[
            (full_dev_df["subject_id"] == subj) & (full_dev_df["date"] <= target_date)
        ].sort_values("date")

        if len(hist) < seq_len:
            pad = pd.DataFrame(0, index=range(seq_len - len(hist)), columns=hist.columns)
            hist = pd.concat([pad, hist], ignore_index=True)
        hist = hist.tail(seq_len)

        x = np.clip(
            np.nan_to_num(
                hist[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            ),
            -10,
            10,
        )
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        probs_list = []
        with torch.no_grad():
            for m in models:
                probs_list.append(torch.sigmoid(m(x_tensor)).cpu().numpy()[0])
        pred_rows.append(np.mean(probs_list, axis=0))

    pred_array = np.vstack(pred_rows) * CALIBRATION_SCALE + CALIBRATION_SHIFT
    submission[label_cols] = pred_array

    out_path = out_path or os.path.join(OUT_DIR, SUBMISSION_NAME)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"submission saved: {out_path}")
    print(submission[label_cols].describe())
    return submission
