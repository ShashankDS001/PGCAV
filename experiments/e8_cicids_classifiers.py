"""
Experiment 8 — CIC-IDS-2017 Classifier Training
================================================
Trains RF and MLP on the Mon-Thu split (full dataset), evaluates on
the Fri split, and saves models to models/cicids/.

Key differences from e4 (UNSW-NB15):
* CIC-IDS-2017 has no string categorical columns — all features numeric
* Target column is 'label_binary' (0=benign, 1=attack)
* Training set is ~1.87M rows (larger than UNSW-NB15 Format B)
* Training class imbalance: ~86% benign, ~14% attack

Outputs:
    models/cicids/preprocessor.pkl
    models/cicids/rf.pkl
    models/cicids/mlp.pkl
    results/tables/cicids_classifier_baseline.csv

Run:
    python -m experiments.e8_cicids_classifiers
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import CICIDS_MODELS_DIR, RESULTS_TABLES
from grammar.data_loader_cicids import load_cicids_train_test
from pipeline.preprocessor import Preprocessor
from pipeline.classifiers import (
    train_rf, train_mlp, save_model, evaluate_model, metrics_table,
)

# CIC-IDS-2017 specific preprocessor settings
_CICIDS_DROP  = ["label", "label_binary"]  # drop string label; target = label_binary
_CICIDS_CATS  = []                          # all features are numeric post-CICFlowMeter


def main() -> Path:
    print("=" * 70)
    print("E8 — CIC-IDS-2017 classifier training (full dataset)")
    print("=" * 70)
    overall_t0 = time.time()

    # ── Step 1: load ───────────────────────────────────────────────
    print("\n[1/4] Loading Mon-Thu (train) / Fri (test) splits ...")
    train_df, test_df = load_cicids_train_test(use_cache=True)
    print(f"      train {train_df.shape}  "
          f"benign={int((train_df['label_binary']==0).sum()):,}  "
          f"attack={int((train_df['label_binary']==1).sum()):,}")
    print(f"      test  {test_df.shape}  "
          f"benign={int((test_df['label_binary']==0).sum()):,}  "
          f"attack={int((test_df['label_binary']==1).sum()):,}")

    # ── Step 2: preprocess ─────────────────────────────────────────
    print("\n[2/4] Fitting preprocessor ...")
    pre = Preprocessor(
        target_col="label_binary",
        drop_cols=_CICIDS_DROP,
        categorical_cols=_CICIDS_CATS,
    )
    X_train, y_train = pre.fit_transform(train_df)
    X_test,  y_test  = pre.transform(test_df)
    print(f"      X_train: {X_train.shape}   X_test: {X_test.shape}")
    pre_path = pre.save(CICIDS_MODELS_DIR / "preprocessor.pkl")
    print(f"      saved preprocessor → {pre_path}")

    # ── Step 3: train ──────────────────────────────────────────────
    print("\n[3/4] Training classifiers ...")
    rf = train_rf(X_train, y_train)
    save_model(rf, "rf", models_dir=CICIDS_MODELS_DIR)

    mlp = train_mlp(X_train, y_train)
    save_model(mlp, "mlp", models_dir=CICIDS_MODELS_DIR)

    # ── Step 4: evaluate ───────────────────────────────────────────
    print("\n[4/4] Evaluating on Friday test split ...")
    rows = [
        evaluate_model(rf,  X_test, y_test, name="CIC_RandomForest"),
        evaluate_model(mlp, X_test, y_test, name="CIC_MLP"),
    ]
    table = metrics_table(rows)
    out = RESULTS_TABLES / "cicids_classifier_baseline.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved baseline metrics → {out}")

    print("\n" + "=" * 90)
    print(f"{'model':<20}{'accuracy':>10}{'precision':>11}{'recall':>9}"
          f"{'f1':>8}{'roc_auc':>10}{'tpr':>8}{'fpr':>8}")
    print("=" * 90)
    for _, r in table.iterrows():
        print(f"{r['model']:<20}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
              f"{r['recall']:>9.4f}{r['f1']:>8.4f}{r['roc_auc']:>10.4f}"
              f"{r['tpr']:>8.4f}{r['fpr']:>8.4f}")
    print("=" * 90)

    for _, r in table.iterrows():
        if r["accuracy"] > 0.999:
            print(f"\n  WARNING: {r['model']} acc={r['accuracy']:.4f} — check for label leakage")
        if r["accuracy"] < 0.80:
            print(f"\n  NOTE: {r['model']} acc={r['accuracy']:.4f} — "
                  f"CIC-IDS-2017 train/test class imbalance may affect threshold")

    print(f"\nTotal: {time.time()-overall_t0:.1f}s")
    return out


if __name__ == "__main__":
    main()
