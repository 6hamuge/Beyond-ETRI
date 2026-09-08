"""Training / evaluation loops and Leave-One-Subject-Out cross-validation."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import CONFIG, LABEL_COLS
from .dataset import LifelogDataset, feature_order
from .model import PTDTransformer


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(
    y_true: np.ndarray,
    y_pred_logit: np.ndarray,
    label_cols: list[str] = LABEL_COLS,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Per-label Accuracy / F1 / AUC plus their macro averages."""
    y_prob = torch.sigmoid(torch.tensor(y_pred_logit)).numpy()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = np.array(y_true)

    metrics: dict[str, dict[str, float]] = {}
    for i, label in enumerate(label_cols):
        y_label, pred_label, prob_label = y_true[:, i], y_pred[:, i], y_prob[:, i]
        mask = ~np.isnan(y_label)
        if mask.sum() == 0:
            continue
        metrics[label] = {
            "acc": accuracy_score(y_label[mask], pred_label[mask]),
            "f1": f1_score(y_label[mask], pred_label[mask], zero_division=0),
        }
        try:
            metrics[label]["auc"] = roc_auc_score(y_label[mask], prob_label[mask])
        except ValueError:
            metrics[label]["auc"] = 0.5

    if metrics:
        metrics["macro"] = {
            "acc": float(np.mean([v["acc"] for v in metrics.values()])),
            "f1": float(np.mean([v["f1"] for v in metrics.values()])),
            "auc": float(np.mean([v["auc"] for v in metrics.values()])),
        }
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_cols: list[str] = LABEL_COLS,
) -> dict[str, dict[str, float]]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu().numpy())
        all_labels.append(y.numpy())
    return compute_metrics(np.vstack(all_labels), np.vstack(all_logits), label_cols)


def run_loso_cv(
    full_dev_df,
    raw_cols: list[str],
    dev_cols: list[str],
    label_cols: list[str] = LABEL_COLS,
    config: dict = CONFIG,
    device: torch.device | None = None,
) -> dict:
    """Leave-One-Subject-Out CV.

    For each held-out test subject, one other subject (chosen cyclically) is the
    validation set for early stopping on macro AUC; the best checkpoint is then
    scored on the test subject.
    """
    device = device or get_device()
    feat_cols = feature_order(raw_cols, dev_cols)
    n_features = len(feat_cols)
    n_labels = len(label_cols)

    all_subjects = sorted(full_dev_df["subject_id"].unique())
    loso_results: dict = {}

    for test_subj in all_subjects:
        test_df = full_dev_df[full_dev_df["subject_id"] == test_subj]
        trainval_df = full_dev_df[full_dev_df["subject_id"] != test_subj]

        candidates = [s for s in all_subjects if s != test_subj]
        val_subj = candidates[all_subjects.index(test_subj) % len(candidates)]
        train_df = trainval_df[trainval_df["subject_id"] != val_subj]
        val_df = trainval_df[trainval_df["subject_id"] == val_subj]

        train_ds = LifelogDataset(train_df, feat_cols, label_cols, config["seq_len"])
        val_ds = LifelogDataset(val_df, feat_cols, label_cols, config["seq_len"])
        test_ds = LifelogDataset(test_df, feat_cols, label_cols, config["seq_len"])
        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            continue

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

        model = PTDTransformer(
            n_features=n_features,
            n_labels=n_labels,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            seq_len=config["seq_len"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"]
        )
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None
        patience_count = 0
        history = {"train_loss": [], "val_auc": []}

        for _epoch in range(1, config["epochs"] + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics = evaluate(model, val_loader, device, label_cols)
            scheduler.step()

            val_auc = val_metrics.get("macro", {}).get("auc", 0.5)
            history["train_loss"].append(train_loss)
            history["val_auc"].append(val_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= config["patience"]:
                break

        if best_state is None:
            continue
        model.load_state_dict(best_state)
        loso_results[test_subj] = {
            "metrics": evaluate(model, test_loader, device, label_cols),
            "history": history,
        }

    print(f"LOSO-CV completed: {len(loso_results)}/{len(all_subjects)} subjects")
    return loso_results


def summarize_loso(
    loso_results: dict, label_cols: list[str] = LABEL_COLS
) -> "np.ndarray":
    """Print the mean +/- std Accuracy / F1 / AUC table across folds."""
    summary = {lbl: {"acc": [], "f1": [], "auc": []} for lbl in [*label_cols, "macro"]}
    for result in loso_results.values():
        for lbl, m in result["metrics"].items():
            if lbl in summary:
                summary[lbl]["acc"].append(m.get("acc", 0))
                summary[lbl]["f1"].append(m.get("f1", 0))
                summary[lbl]["auc"].append(m.get("auc", 0.5))

    print("\nLOSO-CV results")
    print(f'{"label":8s}  {"Acc":>13s}  {"F1":>13s}  {"AUC":>13s}')
    print("-" * 54)
    for lbl in [*label_cols, "macro"]:
        if summary[lbl]["auc"]:
            print(
                f"{lbl:8s}  "
                f'{np.mean(summary[lbl]["acc"]):.3f}±{np.std(summary[lbl]["acc"]):.3f}  '
                f'{np.mean(summary[lbl]["f1"]):.3f}±{np.std(summary[lbl]["f1"]):.3f}  '
                f'{np.mean(summary[lbl]["auc"]):.3f}±{np.std(summary[lbl]["auc"]):.3f}'
            )
    return summary
