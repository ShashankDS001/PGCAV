"""
PGCAV Classifiers
=================
Train, save, load, and evaluate the baseline classifiers used in Paper 2.

Choices and rationale:
* RandomForest    — strong tabular baseline, no gradient (forces black-box
                    attacks like HopSkipJump). Most NIDS papers report on it.
* MLPClassifier   — differentiable, lets ART run white-box attacks
                    (FGSM, PGD, C&W) directly.

We deliberately keep both inside sklearn so there is one dependency set
and the pickled artefacts are portable. PyTorch is introduced only in
Phase 2C when we wire up ART for adversarial generation.
"""

from __future__ import annotations
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config import RF_PARAMS, MLP_PARAMS, MODELS_DIR, VERBOSE


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[clf] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_rf(X: np.ndarray, y: np.ndarray, **overrides) -> RandomForestClassifier:
    """Train a RandomForest binary classifier on (X, y)."""
    params = {**RF_PARAMS, **overrides}
    _log(f"training RF  n={len(X):,}  d={X.shape[1]}  params={params}")
    t0 = time.time()
    clf = RandomForestClassifier(**params)
    clf.fit(X, y)
    _log(f"RF trained in {time.time()-t0:.1f}s")
    return clf


def train_mlp(X: np.ndarray, y: np.ndarray, **overrides) -> MLPClassifier:
    """Train an MLPClassifier on (X, y)."""
    params = {**MLP_PARAMS, **overrides}
    _log(f"training MLP  n={len(X):,}  d={X.shape[1]}  params={params}")
    t0 = time.time()
    clf = MLPClassifier(**params)
    clf.fit(X, y)
    _log(f"MLP trained in {time.time()-t0:.1f}s  "
         f"final_loss={getattr(clf, 'loss_', float('nan')):.4f}  "
         f"iters={getattr(clf, 'n_iter_', -1)}")
    return clf


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────

def save_model(model: Any, name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Pickle a model under models/{name}.pkl."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    _log(f"saved {name} to {path}")
    return path


def load_model(name: str, models_dir: Path = MODELS_DIR) -> Any:
    """Load a previously saved model."""
    path = models_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "model",
) -> dict:
    """Compute the standard binary-classifier metric suite."""
    t0 = time.time()
    y_pred = model.predict(X_test)
    # roc_auc needs probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # edge case: only one class present
        tn = fp = fn = tp = 0

    try:
        auc = float(roc_auc_score(y_test, y_score))
    except ValueError:
        auc = float("nan")  # only one class in y_test

    metrics = {
        "model": name,
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "eval_seconds": round(time.time() - t0, 2),
    }
    return metrics


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Format a list of `evaluate_model` results as a CSV-ready DataFrame."""
    cols = [
        "model", "n_test", "accuracy", "precision", "recall", "f1",
        "roc_auc", "tpr", "fpr", "tp", "fp", "tn", "fn", "eval_seconds",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]
