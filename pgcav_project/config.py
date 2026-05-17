"""
PGCAV Paper 1 — Configuration
==============================
Single source of truth for paths, seeds, and thresholds.
All scripts import from this file. Never hardcode paths elsewhere.
"""

from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Paths (resolved relative to this file) ────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_TABLES = BASE_DIR / "results" / "tables"
RESULTS_FIGS = BASE_DIR / "results" / "figures"

# Make sure output dirs exist at import time
for _d in [DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGS]:
    _d.mkdir(parents=True, exist_ok=True)

# ── UNSW-NB15 dataset files ───────────────────────────────────────────
# Two formats are supported. The loader auto-detects which one is present.
#
# Format A — original 4-CSV raw capture (~2.54M rows, 49 features, no header):
#   Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
UNSW_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]
UNSW_TRAIN_FILES = UNSW_FILES[:3]      # files 1-3 = training split
UNSW_TEST_FILE = [UNSW_FILES[3]]       # file 4   = test split (held out)
UNSW_FEATURES_FILE = "NUSW-NB15_features.csv"  # column-name reference

# Format B — preprocessed train/test split (~257k rows, 45 features, parquet):
#   Already has headers, pre-split, balanced for ML benchmarking.
#   This is what most ML-IDS papers use as the standard benchmark.
#   Filenames as published on the UNSW research site / Kaggle:
UNSW_PARQUET_TRAIN = "UNSW_NB15_training-set.parquet"
UNSW_PARQUET_TEST = "UNSW_NB15_testing-set.parquet"
# Also accept the CSV variants of the preprocessed version:
UNSW_PREPROC_CSV_TRAIN = "UNSW_NB15_training-set.csv"
UNSW_PREPROC_CSV_TEST = "UNSW_NB15_testing-set.csv"

# Cached intermediate (Parquet, ~10x faster to reload than re-parsing CSVs)
PROCESSED_NORMAL_PARQUET = DATA_PROCESSED / "unsw_normal.parquet"
PROCESSED_FULL_PARQUET = DATA_PROCESSED / "unsw_full.parquet"

# ── Column name conventions ───────────────────────────────────────────
# UNSW-NB15 CSVs have NO HEADER and use mixed capitalisation
# (e.g. "Spkts" not "spkts", "Sjit" not "sjit"). We normalise everything
# to lowercase at load time so constraints can use predictable names.
LABEL_COL = "label"          # 0 = normal, 1 = attack
ATTACK_COL = "attack_cat"

# ── Grammar calibration thresholds ────────────────────────────────────
GRAMMAR_FPR_ACTIVE_LIMIT = 0.02    # 2%  — Tier 1 active constraint must stay below this
GRAMMAR_FPR_TIER2_LIMIT = 0.05     # 5%  — above this, demote to informational Tier 2
BOOTSTRAP_RESAMPLES = 1000         # for 95% CIs on per-constraint FPR
BOOTSTRAP_SAMPLE_SIZE = 100_000    # subsample size for bootstrap (full would be slow)

# ── Tolerance for floating-point Tier 2 constraints ───────────────────
SMEAN_REL_TOLERANCE = 0.05         # 5% relative tolerance on smean = sbytes/spkts
TCPRTT_ABS_TOLERANCE = 0.01        # 10 ms absolute tolerance on tcprtt = synack + ackdat

# ── Phase 2: Classifier training paths ────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
DATA_ADVERSARIAL = BASE_DIR / "data" / "adversarial"
for _d in [MODELS_DIR, DATA_ADVERSARIAL]:
    _d.mkdir(parents=True, exist_ok=True)

# Train/test split convention from the UNSW-NB15 paper:
#   Files 1-3 = training, File 4 = test (held out)
PROCESSED_TRAIN_PARQUET = DATA_PROCESSED / "unsw_train.parquet"
PROCESSED_TEST_PARQUET = DATA_PROCESSED / "unsw_test.parquet"

# Columns dropped from the feature matrix before training
# (target leakage, identifiers, timestamps, raw IPs).
DROP_COLS_FOR_TRAINING = [
    "srcip", "dstip",                # raw IPs — too high cardinality
    "stime", "ltime",                # timestamps — leak via session ordering
    "attack_cat",                    # multi-class label, would leak binary target
    "label",                         # the target itself, separated explicitly
]

# Categorical columns — one-hot encoded
CATEGORICAL_COLS = ["proto", "service", "state"]

# Classifier hyperparameters (kept conservative for reproducibility)
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 256,
    "learning_rate_init": 1e-3,
    "max_iter": 50,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 5,
    "random_state": RANDOM_SEED,
    "verbose": False,
}

# ── Logging verbosity ─────────────────────────────────────────────────
VERBOSE = True
