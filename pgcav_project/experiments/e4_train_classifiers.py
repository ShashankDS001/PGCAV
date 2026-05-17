"""
Experiment 4 — Classifier training and baseline evaluation
===========================================================
Trains RF and MLP on the official UNSW-NB15 train split (files 1-3),
evaluates on the official test split (file 4), and saves both models
plus the fitted preprocessor for later use by the adversarial-generation
script (Phase 2C).

Outputs:
    models/preprocessor.pkl
    models/rf.pkl
    models/mlp.pkl
    results/tables/classifier_baseline.csv

Run:
    python -m experiments.e4_train_classifiers
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import MODELS_DIR, RESULTS_TABLES
from pipeline.preprocessor import Preprocessor, build_train_test_split
from pipeline.classifiers import (
    train_rf, train_mlp, save_model, evaluate_model, metrics_table,
)


def main() -> Path:
    print("=" * 70)
    print("E4 — Classifier training and baseline evaluation")
    print("=" * 70)

    # ── Step 1: load official train/test split ──────────────────────
    overall_t0 = time.time()
    print("\n[1/4] Loading train (files 1-3) and test (file 4) splits ...")
    train_df, test_df = build_train_test_split(use_cache=True)
    print(f"      train shape: {train_df.shape}   "
          f"normal={int((train_df['label']==0).sum()):,}   "
          f"attack={int((train_df['label']==1).sum()):,}")
    print(f"      test  shape: {test_df.shape}    "
          f"normal={int((test_df['label']==0).sum()):,}   "
          f"attack={int((test_df['label']==1).sum()):,}")

    # ── Step 2: fit preprocessor on train, transform both ───────────
    print("\n[2/4] Fitting preprocessor on train, transforming both splits ...")
    pre = Preprocessor()
    X_train, y_train = pre.fit_transform(train_df)
    X_test, y_test = pre.transform(test_df)
    print(f"      X_train: {X_train.shape}   X_test: {X_test.shape}")
    pre_path = pre.save(MODELS_DIR / "preprocessor.pkl")

    # ── Step 3: train RF + MLP ──────────────────────────────────────
    print("\n[3/4] Training classifiers ...")
    rf = train_rf(X_train, y_train)
    save_model(rf, "rf")
    mlp = train_mlp(X_train, y_train)
    save_model(mlp, "mlp")

    # ── Step 4: evaluate both on the held-out test split ────────────
    print("\n[4/4] Evaluating on UNSW-NB15 test split (file 4) ...")
    rows = [
        evaluate_model(rf, X_test, y_test, name="RandomForest"),
        evaluate_model(mlp, X_test, y_test, name="MLP"),
    ]
    table = metrics_table(rows)

    out_path = RESULTS_TABLES / "classifier_baseline.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved baseline metrics to: {out_path}")

    # Pretty-print
    print("\n" + "=" * 90)
    print(f"{'model':<14}{'accuracy':>10}{'precision':>11}{'recall':>9}"
          f"{'f1':>8}{'roc_auc':>10}{'tpr':>8}{'fpr':>8}")
    print("=" * 90)
    for _, r in table.iterrows():
        print(f"{r['model']:<14}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
              f"{r['recall']:>9.4f}{r['f1']:>8.4f}{r['roc_auc']:>10.4f}"
              f"{r['tpr']:>8.4f}{r['fpr']:>8.4f}")
    print("=" * 90)

    # ── Sanity check warnings ───────────────────────────────────────
    for _, r in table.iterrows():
        if r["accuracy"] > 0.999:
            print(f"\n  WARNING: {r['model']} accuracy = {r['accuracy']:.4f}")
            print(f"    Suspiciously high. Check for target leakage in features.")
            print(f"    UNSW-NB15 typically yields 0.95-0.99 accuracy for these models.")
        if r["accuracy"] < 0.80:
            print(f"\n  WARNING: {r['model']} accuracy = {r['accuracy']:.4f}")
            print(f"    Lower than UNSW-NB15 norm. Check preprocessing or class imbalance.")

    print(f"\nTotal time: {time.time()-overall_t0:.1f}s")
    return out_path


if __name__ == "__main__":
    main()
