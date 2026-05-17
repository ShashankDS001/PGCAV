# PGCAV Paper 1 + Phase 2A — Protocol Grammar & Classifier Baseline (UNSW-NB15)

Implementation of the 12-constraint Protocol Grammar (Paper 1) plus
classifier training pipeline (Paper 2 preparation), calibrated on UNSW-NB15.

## Quick start

```bash
# 1. Set up a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify on synthetic data BEFORE downloading anything
python tests/test_constraints.py
python tests/test_dataloader.py
python tests/test_pipeline.py
python tests/test_end_to_end.py

# 4. Download UNSW-NB15 into data/raw/  (EITHER format works)
#    The loader auto-detects which version you have:
#
#    Format A — original raw 4-CSV capture (~2.54M rows):
#       UNSW-NB15_1.csv ... UNSW-NB15_4.csv + NUSW-NB15_features.csv
#    Format B — preprocessed train/test split (~257k rows, what most papers use):
#       UNSW_NB15_training-set.parquet  (or .csv)
#       UNSW_NB15_testing-set.parquet   (or .csv)
#
#    Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

# 5. Run the full pipeline
python run_pipeline.py
```

`run_pipeline.py` orchestrates four steps:

| Step          | Script                              | Output                                  |
|---------------|-------------------------------------|------------------------------------------|
| `eda`         | experiments/e1_eda.py               | results/tables/eda_feature_stats.csv    |
| `calibration` | experiments/e2_calibration.py       | results/tables/calibration_table.csv    |
| `figures`     | experiments/e3_figures.py           | results/figures/fig{1,2,3}_*.png        |
| `classifiers` | experiments/e4_train_classifiers.py | models/{preprocessor,rf,mlp}.pkl + results/tables/classifier_baseline.csv |

Run a single step:
```bash
python run_pipeline.py --only calibration
```

Skip a step (e.g. after first run):
```bash
python run_pipeline.py --skip eda --skip figures
```

## What lands where

```
pgcav_project/
├── config.py                        # paths, seeds, hyperparameters
├── run_pipeline.py                  # full pipeline driver (Phase 1 + 2A)
├── run_paper1.py                    # Phase 1 only driver (legacy)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                         # YOU place UNSW-NB15 CSVs here
│   ├── processed/                   # Parquet cache (auto-generated)
│   └── adversarial/                 # Phase 2C output (not yet implemented)
├── grammar/
│   ├── data_loader.py               # UNSW-NB15 ingestion + cleanup
│   ├── constraints.py               # 12 vectorised constraints
│   └── validator.py                 # validate_row / validate_dataframe / calibrate_constraints
├── pipeline/                        # Phase 2A
│   ├── preprocessor.py              # one-hot + scaling, with inverse transform
│   └── classifiers.py               # train_rf / train_mlp / evaluate_model
├── experiments/
│   ├── e1_eda.py
│   ├── e2_calibration.py
│   ├── e3_figures.py
│   └── e4_train_classifiers.py
├── models/                          # trained models + fitted preprocessor
├── results/
│   ├── tables/                      # all CSV outputs land here
│   └── figures/                     # PNG figures at 300 DPI
└── tests/
    ├── test_constraints.py          # 14 constraint unit tests
    ├── test_dataloader.py           # loader cleanup tests
    ├── test_end_to_end.py           # full Phase 1 pipeline on synthetic CSV
    └── test_pipeline.py             # Phase 2A pipeline on synthetic data
```

## The 12 constraints

| ID  | Tier | Source       | Rule                                       |
|-----|------|--------------|--------------------------------------------|
| C01 | 1    | RFC 791 §3.1 | IP TTL ∈ [1, 255]                          |
| C02 | 1    | RFC 793 §3.1 | TCP window ∈ [0, 65535] when proto=tcp     |
| C03 | 1    | RFC 793 §3.1 | TCP base-seq ∈ [0, 2³²−1]                  |
| C04 | 1    | Derived      | flow duration ≥ 0                          |
| C12 | 1    | Derived      | jitter magnitudes ≥ 0                      |
| C05 | 2    | Derived      | tcprtt, synack, ackdat ≥ 0                 |
| C06 | 2    | Derived      | sbytes, dbytes ≥ 0                         |
| C07 | 2    | Derived      | spkts, dpkts ≥ 0                           |
| C08 | 2    | Derived      | sbytes ≥ spkts and dbytes ≥ dpkts          |
| C09 | 2    | Derived      | sloss ≤ spkts and dloss ≤ dpkts            |
| C10 | 2    | Derived      | smeansz · spkts ≈ sbytes (within tol.)     |
| C11 | 2    | Derived      | TCP: tcprtt ≈ synack + ackdat              |

Tier 1 = RFC-direct. Tier 2 = logical invariant derived from the UNSW-NB15
feature definitions.

## Go / No-Go gate

`e2_calibration.py` prints the gate result. Paper 1 is ready to write when:

* mean FPR across **Active** constraints ≤ **2.00%**
* no single constraint exceeds **5.00%**

If the gate fails, the calibration table tells you which constraint to widen.
The two most likely failures and their fixes:

* **C10 smean_consistency** — increase `SMEAN_REL_TOLERANCE` in `config.py`.
  UNSW-NB15 rounds `smeansz` to integer, so 5% tolerance is the floor.
* **C11 tcp_handshake_timing** — increase `TCPRTT_ABS_TOLERANCE` in `config.py`.
  Timing fields are floating-point with non-trivial rounding error.

Do NOT relax C01–C04 (Tier 1). If those fail, the bug is in your data load.

## Reproducibility

* Random seed pinned in `config.py` (`RANDOM_SEED = 42`)
* Bootstrap resamples and sample size also pinned
* Dependencies pinned to exact versions in `requirements.txt`
* Parquet cache makes re-runs idempotent

For top-venue artifact evaluation, also publish a Docker image — not included
here but trivial to add (the Dockerfile would just `pip install -r requirements.txt`).

## What this does NOT include yet

You now have **Phase 1 (grammar + calibration) + Phase 2A (classifier baseline)**.
The remaining pieces for the merged Path C top-venue paper are:

* **Phase 2C — Adversarial generation (next deliverable)** — ART library, FGSM/PGD/C&W/HopSkipJump
* **Phase 2D — Apply grammar to adversarials** — measure violation rate (VR) and Effective Threat Rate (ETR)
* **Phase 3 — Detection layer** — use grammar violation as a runtime forensic detector
* **Phase 4 — Cross-dataset validation** — CIC-IDS-2017 and TON_IoT loaders
* **Phase 5 — Adaptive constraint-aware adversary** — projected-gradient attacks that respect the grammar

## Citing

If you build on this code, please cite the eventual paper. Until then:

```bibtex
@misc{pgcav_grammar_2026,
  title = {A Formal Protocol Constraint Grammar for NetFlow Intrusion Detection},
  author = {Your Name},
  year = {2026},
  note = {Implementation of the PGCAV grammar over UNSW-NB15}
}
```
