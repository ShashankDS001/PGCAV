"""
PGCAV Adversarial Generation  (Phase 2C — C2 Detection framing)
================================================================
Four attacks, all compatible with sklearn classifiers:

    Random-Uniform   — ε-bounded uniform noise (baseline)
    FGSM             — single-step finite-difference gradient
    PGD              — iterative FGSM (multi-step)
    HopSkipJump      — ART decision-based black-box (label-only queries)

WHY FINITE DIFFERENCES FOR FGSM/PGD
-------------------------------------
ART 1.18+ removed LossGradientsMixin from ScikitlearnClassifier, so
gradient-based ART attacks (FastGradientMethod, PGD) no longer work
directly against sklearn models.  We estimate the sign-gradient by
finite-differencing predict_proba — mathematically equivalent to
a black-box gradient FGSM and accepted practice in the literature.

For n=1000 attack flows and d=76 features, this is ~152k predict_proba
calls (fast for both RF and MLP).

INSTALL
-------
    pip install adversarial-robustness-toolbox==1.18.0
    (only needed for HopSkipJump)
"""

from __future__ import annotations
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from config import VERBOSE

# ART import guard — only HopSkipJump uses ART
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from art.estimators.classification.scikitlearn import ScikitlearnClassifier
        from art.attacks.evasion import HopSkipJump as _ArtHSJ
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[adv] {msg}", flush=True)


def _require_art() -> None:
    if not ART_AVAILABLE:
        raise ImportError(
            "\n  ART not installed.  Run:\n"
            "    pip install adversarial-robustness-toolbox==1.18.0\n"
        )


# ──────────────────────────────────────────────────────────────────────
# Gradient estimation (finite differences on predict_proba)
# ──────────────────────────────────────────────────────────────────────

def _sign_gradient(
    model,
    X: np.ndarray,
    attack_class: int = 1,
    eps_fd: float = 1e-4,
    batch_size: int = 128,
) -> np.ndarray:
    """Estimate sign(∂L/∂x) via finite differences on predict_proba.

    L = probability of attack_class. We want to DECREASE this (evasion),
    so the adversarial perturbation is in the -sign(gradient) direction.

    Parameters
    ----------
    model        : sklearn classifier with predict_proba()
    X            : input batch, shape (n, d)
    attack_class : class we want to evade detection of (1 = attack)
    eps_fd       : finite-difference step size
    batch_size   : rows processed per predict call

    Returns
    -------
    sign_grad : np.ndarray shape (n, d), values in {-1, 0, 1}
    """
    n, d = X.shape
    sign_grad = np.zeros((n, d), dtype=np.float32)

    for batch_start in range(0, n, batch_size):
        sl = slice(batch_start, min(batch_start + batch_size, n))
        X_b = X[sl].astype(np.float64)
        n_b = X_b.shape[0]
        col_grads = np.zeros((n_b, d), dtype=np.float64)

        for j in range(d):
            X_plus = X_b.copy();  X_plus[:, j] += eps_fd
            X_minus = X_b.copy(); X_minus[:, j] -= eps_fd
            p_plus  = model.predict_proba(X_plus.astype(np.float32))[:, attack_class]
            p_minus = model.predict_proba(X_minus.astype(np.float32))[:, attack_class]
            col_grads[:, j] = (p_plus - p_minus) / (2.0 * eps_fd)

        sign_grad[sl] = np.sign(col_grads).astype(np.float32)

    return sign_grad


# ──────────────────────────────────────────────────────────────────────
# Attack generators
# ──────────────────────────────────────────────────────────────────────

def generate_random(
    X: np.ndarray,
    eps: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Random uniform ε-perturbation — baseline attack."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-eps, eps, X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def generate_fgsm(
    model,
    X: np.ndarray,
    eps: float = 0.1,
    attack_class: int = 1,
    batch_size: int = 128,
) -> np.ndarray:
    """FGSM via finite-difference gradient estimation.

    Works with any sklearn classifier that implements predict_proba.
    Perturbation is in the -sign(∇p_attack) direction.
    """
    _log(f"FGSM  eps={eps}  n={len(X):,}")
    sg = _sign_gradient(model, X, attack_class=attack_class, batch_size=batch_size)
    return (X - eps * sg).astype(np.float32)


def generate_pgd(
    model,
    X: np.ndarray,
    eps: float = 0.1,
    eps_step: Optional[float] = None,
    max_iter: int = 10,
    attack_class: int = 1,
    batch_size: int = 128,
) -> np.ndarray:
    """PGD: iterative FGSM with ε-ball projection.

    Works with any sklearn classifier that implements predict_proba.
    """
    if eps_step is None:
        eps_step = eps / max_iter
    _log(f"PGD  eps={eps}  step={eps_step:.4f}  iters={max_iter}  n={len(X):,}")
    X_adv = X.copy().astype(np.float32)
    X_orig = X.copy().astype(np.float32)
    for i in range(max_iter):
        sg = _sign_gradient(model, X_adv, attack_class=attack_class,
                            batch_size=batch_size)
        X_adv = X_adv - eps_step * sg
        # Project back to ε-ball around X_orig
        delta = np.clip(X_adv - X_orig, -eps, eps)
        X_adv = (X_orig + delta).astype(np.float32)
    return X_adv


def generate_hopskipjump(
    model,
    X: np.ndarray,
    max_eval: int = 100,
    max_iter: int = 20,
    batch_size: int = 4,
) -> np.ndarray:
    """HopSkipJump via ART — decision-based, no gradient needed.

    Works with RF and MLP.  Slow: budget max_eval queries per sample.
    Limit X to 200-500 rows.
    """
    _require_art()
    _log(f"HSJ  max_eval={max_eval}  max_iter={max_iter}  n={len(X):,}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        art_clf = ScikitlearnClassifier(model=model, clip_values=None)
    attack = _ArtHSJ(
        classifier=art_clf,
        targeted=False,
        max_eval=max_eval,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=False,
    )
    return attack.generate(X)


# ──────────────────────────────────────────────────────────────────────
# Feature-space reconstruction for grammar validation
# ──────────────────────────────────────────────────────────────────────

def reconstruct_for_grammar(
    X_adv: np.ndarray,
    preprocessor,
    df_orig: pd.DataFrame,
) -> pd.DataFrame:
    """Map adversarials from scaled feature space to original scale.

    Inverse-scales numeric features; reattaches categorical columns
    (proto, service, state) from the original rows — these are frozen
    because gradient attacks rarely produce coherent protocol changes
    through one-hot encoding.
    """
    df_num_inv = preprocessor.inverse_numeric(X_adv)
    for col in ("proto", "service", "state"):
        if col in df_orig.columns:
            df_num_inv[col] = df_orig[col].values[:len(X_adv)]
    return df_num_inv


# ──────────────────────────────────────────────────────────────────────
# Detection metrics  (the C2 contribution)
# ──────────────────────────────────────────────────────────────────────

def compute_detection_metrics(
    y_clf_pred: np.ndarray,
    grammar_valid: np.ndarray,
    attack_label: int = 1,
) -> dict:
    """Compute the C2 detection metric suite.

    Parameters
    ----------
    y_clf_pred    : classifier predictions on adversarials (0=benign, 1=attack)
    grammar_valid : True if grammar says flow is valid (no violation)
    attack_label  : true label of all inputs (1 = attack class)

    Returns
    -------
    dict with keys:
        n_total, n_evading, evasion_rate (ER)
        n_viol_all, vr_all
        n_viol_evading, vr_evading, dr
        combined_tpr   = 1 - ER*(1 - VR_evading)
        etr            = ER*(1 - VR_evading)
        clf_only_tpr   = 1 - ER
    """
    n = len(y_clf_pred)
    assert len(grammar_valid) == n

    evading_mask = (y_clf_pred != attack_label)
    n_evading = int(evading_mask.sum())
    er = n_evading / n if n else 0.0

    violated = ~np.asarray(grammar_valid, dtype=bool)
    n_viol_all = int(violated.sum())
    vr_all = n_viol_all / n if n else 0.0

    if n_evading > 0:
        n_viol_evading = int(violated[evading_mask].sum())
        vr_evading = n_viol_evading / n_evading
    else:
        n_viol_evading = 0
        vr_evading = 0.0

    combined_tpr = 1.0 - er * (1.0 - vr_evading)
    etr = er * (1.0 - vr_evading)

    return {
        "n_total": n,
        "n_evading": n_evading,
        "evasion_rate": er,
        "n_viol_all": n_viol_all,
        "vr_all": vr_all,
        "n_viol_evading": n_viol_evading,
        "vr_evading": vr_evading,
        "dr": vr_evading,
        "combined_tpr": combined_tpr,
        "etr": etr,
        "clf_only_tpr": 1.0 - er,
    }


def per_constraint_vr(
    grammar_result_df: pd.DataFrame,
    evading_mask: np.ndarray,
    constraint_ids: list[str],
) -> pd.DataFrame:
    """Per-constraint violation rates on all and evading adversarials."""
    evading_mask = np.asarray(evading_mask, dtype=bool)
    n = len(grammar_result_df)
    n_evading = int(evading_mask.sum())
    rows = []
    for cid in constraint_ids:
        if cid not in grammar_result_df.columns:
            continue
        viol = ~grammar_result_df[cid].values.astype(bool)
        vr_a = float(viol.sum() / n) if n else 0.0
        vr_e = float(viol[evading_mask].sum() / n_evading) if n_evading else 0.0
        rows.append({
            "constraint_id": cid,
            "vr_all": vr_a, "vr_all_pct": vr_a * 100.0,
            "vr_evading": vr_e, "vr_evading_pct": vr_e * 100.0,
        })
    return pd.DataFrame(rows).sort_values("vr_all", ascending=False).reset_index(drop=True)
