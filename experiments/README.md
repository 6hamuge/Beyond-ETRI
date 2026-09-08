# `experiments/` — development history

Exploratory notebooks kept for provenance. **None of these is the submission
artifact** — the reproducible pipeline lives in [`../src/`](../src) and
[`../notebooks/PTDTransformer_pipeline.ipynb`](../notebooks/PTDTransformer_pipeline.ipynb).

| Notebook | What it explored |
|----------|------------------|
| `feature_dev_gps_ambience_ble.ipynb` | First pass at nested-list sensor features (GPS speed, ambience scene scores, BLE co-presence). |
| `feature_dev_usage_wifi_hr.ipynb` | App-usage / Wi-Fi / heart-rate bedtime proxies; HR-based sleep-interval estimation. |
| `PTDT_Sleep_Prediction.ipynb` | Initial single PTDTransformer (raw + deviation branches, CLS-token encoder, 7 heads). |
| `PTDT_v3.ipynb` | Dual-tower split (Q vs S), per-label F1-threshold search, 3-/5-day cumulative features. |
| `PTDT_v5_Improved.ipynb` | Adds per-label `pos_weight`, label smoothing, input noise augmentation, subject-quality weighted sampling, feature-attention visualization. |
| `ptdt_xgboost_enhanced_full.ipynb` | XGBoost LOSO-CV baseline with lag / sleep-debt / social-jet-lag features and correlation-based feature pruning. |

## Relationship to the final model

The paper describes a Q/S split (Table II); the **submitted** run that scored
**0.63265** on the leaderboard used the *single* unified PTDTransformer with a
5-seed ensemble and linear probability calibration. That is what `src/` and
`notebooks/PTDTransformer_pipeline.ipynb` reproduce. The dual-tower and XGBoost
notebooks here are ablations / alternatives that did not beat it.
