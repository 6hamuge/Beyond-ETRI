# Beyond-ETRI · PTDTransformer

**A Deviation-Aware Transformer for Multimodal Lifelog-Based Sleep Quality Prediction**

Submission code for the 5th ETRI Human Understanding AI Paper Competition
(제5회 ETRI 휴먼이해 인공지능 논문경진대회). From the lifelogs of 12 smartphone
and wearable sensor streams, the model predicts seven binary daily sleep-related
indicators — Q1–Q3 (subjective: sleep quality, fatigue, stress) and S1–S4
(objective sleep sub-indicators).

- Paper: *PTDTransformer: A Deviation-Aware Transformer for Multimodal
  Lifelog-Based Sleep Quality Prediction* — Chaeyoung Jung, Donghee Kim,
  Yerin Tak (Department of Artificial Intelligence Engineering, Sookmyung
  Women's University)
- Model description: [`docs/MODEL_DESCRIPTION.md`](docs/MODEL_DESCRIPTION.md)
- Leaderboard score: **0.63265** (5-seed ensemble + probability calibration)

## Key idea

Each feature is split into two branches — the subject's **typical value (raw)**
and its **deviation from that subject's own baseline** (a personal z-score
against a trailing 7-day rolling mean/std). The two branches are fused by
`DeviationAwareAttention` (deviation as query, raw behavior as key/value) and
then encoded by a CLS-token Transformer. For the small 10-subject cohort we use
Leave-One-Subject-Out cross-validation (LOSO-CV) plus a multi-seed ensemble and
probability calibration for the final submission.

## Repository layout

```
├── src/                     Modular pipeline (single source of truth)
│   ├── config.py              paths (/data), hyperparameters, seeding
│   ├── data_loading.py        load Parquet sensor tables and labels
│   ├── features/              feature engineering
│   │   ├── common.py            shared utilities + generic daily aggregation
│   │   ├── sleep_proxies.py     screen / light / hr / wifi / usage sleep proxies
│   │   ├── nested_sensors.py    gps / ambience / ble
│   │   ├── composite.py         composite, rolling, and Q1-specific features
│   │   └── deviation.py         label merge + personalized-deviation features
│   ├── dataset.py             sequence dataset (LifelogDataset)
│   ├── model.py               PositionalEncoding, DeviationAwareAttention, PTDTransformer
│   ├── train.py               train / eval loops, LOSO-CV
│   └── predict.py             seed-ensemble training + submission generation
├── scripts/
│   ├── run_loso_cv.py         reproduce the LOSO-CV results
│   └── make_submission.py     reproduce the leaderboard submission file
├── notebooks/
│   └── PTDTransformer_pipeline.ipynb   end-to-end notebook (faithful reproduction)
├── experiments/              development notebooks (Q/S split, XGBoost, ...) — not the submission
├── tests/                    synthetic-data smoke test
├── docs/MODEL_DESCRIPTION.md model description (for submission)
├── data/README.md            /data input layout
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Developed on Google Colab (Python 3.11, NVIDIA T4 or CPU). The model is small
and trains within a few minutes on CPU.

## Data preparation

Place the raw ETRI Lifelog data (12 Parquet files + 2 CSV files) under `/data`.
See [`data/README.md`](data/README.md) for the file list and path convention.
Override the location with environment variables when running locally:

```bash
export DATA_DIR=./data/ch2025
export OUT_DIR=./out
```

## Running

```bash
# Reproduce the LOSO-CV results (paper Table III)
python scripts/run_loso_cv.py

# Reproduce the leaderboard submission -> $OUT_DIR/PTDT_seq7_ensemble_smooth06.csv
python scripts/make_submission.py
```

Or run `notebooks/PTDTransformer_pipeline.ipynb` top to bottom.

## Tests

Run the whole pipeline (features → model → LOSO-CV → submission) on synthetic
data, without needing the real dataset, to check it executes cleanly:

```bash
pip install pytest
pytest -q
```

## LOSO-CV results (paper Table III)

| Label | Accuracy | Macro-F1 | AUC |
|-------|----------|----------|-----|
| Q1 | 0.450 ± 0.166 | 0.386 ± 0.326 | 0.538 ± 0.121 |
| Q2 | 0.489 ± 0.166 | 0.568 ± 0.269 | 0.506 ± 0.112 |
| Q3 | 0.614 ± 0.124 | 0.712 ± 0.181 | 0.494 ± 0.132 |
| S1 | 0.682 ± 0.172 | 0.798 ± 0.125 | 0.482 ± 0.065 |
| S2 | 0.638 ± 0.219 | 0.756 ± 0.173 | 0.433 ± 0.068 |
| S3 | 0.646 ± 0.236 | 0.756 ± 0.201 | 0.524 ± 0.124 |
| S4 | 0.525 ± 0.209 | 0.604 ± 0.279 | 0.497 ± 0.076 |
| **Macro** | 0.578 ± 0.090 | 0.654 ± 0.099 | 0.496 ± 0.046 |

The macro AUC is near chance, reflecting how hard subject-independent
prediction is with only 10 subjects. Ensembling and probability calibration
raised the final leaderboard score to 0.63265. See the model description for
the full analysis.

## Model configuration vs. the paper

The paper's Methodology section describes a Q/S split (a Q-model and an S-model
with separate hyperparameters, Table II). That is an ablation-driven design;
the run that actually produced the **0.63265** leaderboard score used the
*single* unified PTDTransformer (SEQ_LEN 7, d_model 64, 2 layers) with a 5-seed
ensemble and linear probability calibration. That is what `src/` and
`notebooks/PTDTransformer_pipeline.ipynb` reproduce. The dual-tower and XGBoost
notebooks in `experiments/` are alternatives that did not beat it.

## Team

| Name | Affiliation |
|------|-------------|
| Chaeyoung Jung (정채영) | Dept. of Artificial Intelligence Engineering, Sookmyung Women's University |
| Donghee Kim (김동희) | Dept. of Artificial Intelligence Engineering, Sookmyung Women's University |
| Yerin Tak (탁예린) | Dept. of Artificial Intelligence Engineering, Sookmyung Women's University |

Team name `<TEAM_NAME>` · Team number `<TEAM_NUMBER>`
