"""
Experiment 5 — Adversarial corpus generation  (Phase 2C)
=========================================================
Loads the trained classifiers and generates adversarial examples from
attack-class test flows using four attack methods:

    FGSM         — fast gradient (white-box vs MLP)
    PGD          — iterative gradient (white-box vs MLP)
    C&W L2       — optimisation-based (white-box vs MLP, subset only)
    HopSkipJump  — decision-based (black-box vs RF, subset only)

Adversarials are generated in the scaled feature space and saved as
Parquet files under data/adversarial/.  Each file encodes the attack
identity, classifier target, and epsilon in its filename.

C2 framing note: adversarials are UNCONSTRAINED — the attack can
produce any feature value.  Grammar validation happens in e6, not here.

Run:
    python -m experiments.e5_generate_adversarials
or via:
    python run_pipeline.py --only adversarials
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import DATA_ADVERSARIAL, MODELS_DIR, RESULTS_TABLES, VERBOSE
from pipeline.preprocessor import Preprocessor, build_train_test_split
from pipeline.classifiers import load_model
from pipeline.adversarial import (
    ART_AVAILABLE,
    generate_random,
    generate_fgsm,
    generate_pgd,
    generate_hopskipjump,
)


# ── Attack configuration ──────────────────────────────────────────────
# Tune n_samples and epsilons to balance runtime vs coverage.
# FGSM/PGD use finite-difference gradients — fast enough for ~1000 samples.
# HSJ is query-based — limit to 300 samples.
ATTACK_CONFIG = {
    "random": {
        "epsilons": [0.05, 0.1, 0.3],
        "classifiers": ["rf", "mlp"],
    },
    "fgsm": {
        "epsilons": [0.05, 0.1, 0.3],
        "classifiers": ["rf", "mlp"],
        "n_samples": 1000,
    },
    "pgd": {
        "epsilons": [0.1, 0.3],
        "max_iter": 10,
        "classifiers": ["rf", "mlp"],
        "n_samples": 500,
    },
    "hsj": {
        "max_eval": 100,
        "max_iter": 20,
        "classifiers": ["rf", "mlp"],
        "n_samples": 200,
    },
}


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[e5] {msg}", flush=True)


def _adv_path(attack: str, clf_name: str, eps_tag: str) -> Path:
    """Canonical filename for an adversarial corpus."""
    fname = f"adv_{attack}_{clf_name}_{eps_tag}.parquet"
    return DATA_ADVERSARIAL / fname


def _save_adversarials(
    X_orig: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    preprocessor: Preprocessor,
    path: Path,
) -> None:
    """Save adversarial corpus as Parquet with metadata columns."""
    n_num = len(preprocessor.numeric_cols_)
    # Store the full scaled-space adversarial vector plus a few metadata cols
    df = pd.DataFrame(X_adv, columns=preprocessor.feature_names_)
    df["y_true"] = y_true
    # delta L-inf norm
    df["delta_linf"] = np.abs(X_adv - X_orig).max(axis=1)
    df["delta_l2"] = np.linalg.norm(X_adv - X_orig, axis=1)
    df.to_parquet(path, index=False)
    _log(f"  saved {len(df):,} adversarials -> {path.name}")


def main() -> None:
    print("=" * 70)
    print("E5 — Adversarial corpus generation")
    print("=" * 70)
    if not ART_AVAILABLE:
        print("  NOTE: ART not installed — HopSkipJump will be skipped.")
        print("  FGSM, PGD, and Random will still run.")
        print()
    print("=" * 70)

    overall_t0 = time.time()

    # ── Load data ────────────────────────────────────────────────────
    _log("loading train/test split ...")
    train_df, test_df = build_train_test_split(use_cache=True)

    _log("loading preprocessor ...")
    pre_path = MODELS_DIR / "preprocessor.pkl"
    if not pre_path.exists():
        print(f"\n  Preprocessor not found: {pre_path}")
        print("  Run python run_pipeline.py --only classifiers first.")
        return
    preprocessor = Preprocessor.load(pre_path)

    _log("transforming test split ...")
    X_test, y_test = preprocessor.transform(test_df)

    # ── Select attack-class test flows ───────────────────────────────
    # Adversarial attacks perturb ATTACK flows to look BENIGN
    attack_mask = (y_test == 1)
    X_attack = X_test[attack_mask]
    y_attack = y_test[attack_mask]
    # Keep the original test_df rows for later grammar reconstruction
    test_attack_df = test_df[attack_mask.astype(bool)].reset_index(drop=True)
    # Save for e6
    test_attack_df.to_parquet(DATA_ADVERSARIAL / "test_attack_orig.parquet", index=False)
    X_attack_orig_path = DATA_ADVERSARIAL / "test_attack_scaled.npy"
    np.save(X_attack_orig_path, X_attack)

    _log(f"attack-class test flows: {len(X_attack):,}")

    # ── Load classifiers ─────────────────────────────────────────────
    classifiers = {}
    for name in ("rf", "mlp"):
        path = MODELS_DIR / f"{name}.pkl"
        if not path.exists():
            print(f"  WARNING: {name}.pkl not found — skipping attacks that need it")
            continue
        classifiers[name] = load_model(name)

    # ── Summary table rows ────────────────────────────────────────────
    summary_rows = []

    # ── Random ───────────────────────────────────────────────────────
    cfg = ATTACK_CONFIG["random"]
    for clf_name in cfg["classifiers"]:
        if clf_name not in classifiers:
            continue
        clf = classifiers[clf_name]
        for eps in cfg["epsilons"]:
            eps_tag = f"eps{eps:.3f}".replace(".", "p")
            out_path = _adv_path("random", clf_name, eps_tag)
            if out_path.exists():
                continue
            t0 = time.time()
            X_adv = generate_random(X_attack, eps=eps)
            _save_adversarials(X_attack, X_adv, y_attack, preprocessor, out_path)
            er = float((clf.predict(X_adv) != 1).mean())
            summary_rows.append({"attack": "random", "clf": clf_name, "eps": eps,
                                  "n": len(X_adv), "evasion_rate": er,
                                  "time_s": round(time.time()-t0, 1)})
            _log(f"  Random eps={eps}  clf={clf_name}  ER={er:.4f}")

    # ── FGSM ─────────────────────────────────────────────────────────
    cfg = ATTACK_CONFIG["fgsm"]
    n_sub = min(cfg["n_samples"], len(X_attack))
    X_sub = X_attack[:n_sub]; y_sub = y_attack[:n_sub]
    for clf_name in cfg["classifiers"]:
        if clf_name not in classifiers:
            continue
        clf = classifiers[clf_name]
        for eps in cfg["epsilons"]:
            eps_tag = f"eps{eps:.3f}".replace(".", "p")
            out_path = _adv_path("fgsm", clf_name, eps_tag)
            if out_path.exists():
                _log(f"FGSM {clf_name} eps={eps} exists — skipping"); continue
            t0 = time.time()
            X_adv = generate_fgsm(clf, X_sub, eps=eps)
            _save_adversarials(X_sub, X_adv, y_sub, preprocessor, out_path)
            er = float((clf.predict(X_adv) != 1).mean())
            summary_rows.append({"attack": "fgsm", "clf": clf_name, "eps": eps,
                                  "n": n_sub, "evasion_rate": er,
                                  "time_s": round(time.time()-t0, 1)})
            _log(f"  FGSM eps={eps}  clf={clf_name}  ER={er:.4f}  ({time.time()-t0:.0f}s)")

    # ── PGD ──────────────────────────────────────────────────────────
    cfg = ATTACK_CONFIG["pgd"]
    n_sub = min(cfg["n_samples"], len(X_attack))
    X_sub = X_attack[:n_sub]; y_sub = y_attack[:n_sub]
    for clf_name in cfg["classifiers"]:
        if clf_name not in classifiers:
            continue
        clf = classifiers[clf_name]
        for eps in cfg["epsilons"]:
            eps_tag = f"eps{eps:.3f}".replace(".", "p")
            out_path = _adv_path("pgd", clf_name, eps_tag)
            if out_path.exists():
                _log(f"PGD {clf_name} eps={eps} exists — skipping"); continue
            t0 = time.time()
            X_adv = generate_pgd(clf, X_sub, eps=eps, max_iter=cfg["max_iter"])
            _save_adversarials(X_sub, X_adv, y_sub, preprocessor, out_path)
            er = float((clf.predict(X_adv) != 1).mean())
            summary_rows.append({"attack": "pgd", "clf": clf_name, "eps": eps,
                                  "n": n_sub, "evasion_rate": er,
                                  "time_s": round(time.time()-t0, 1)})
            _log(f"  PGD eps={eps}  clf={clf_name}  ER={er:.4f}  ({time.time()-t0:.0f}s)")

    # ── HopSkipJump ──────────────────────────────────────────────────
    cfg = ATTACK_CONFIG["hsj"]
    n_sub = min(cfg["n_samples"], len(X_attack))
    X_sub = X_attack[:n_sub]; y_sub = y_attack[:n_sub]
    for clf_name in cfg["classifiers"]:
        if clf_name not in classifiers:
            continue
        clf = classifiers[clf_name]
        out_path = _adv_path("hsj", clf_name, f"n{n_sub}")
        if out_path.exists():
            _log(f"HSJ {clf_name} exists — skipping"); continue
        if not ART_AVAILABLE:
            _log("ART not installed — skipping HopSkipJump"); continue
        t0 = time.time()
        X_adv = generate_hopskipjump(
            clf, X_sub,
            max_eval=cfg["max_eval"],
            max_iter=cfg["max_iter"],
        )
        _save_adversarials(X_sub, X_adv, y_sub, preprocessor, out_path)
        er = float((clf.predict(X_adv) != 1).mean())
        summary_rows.append({"attack": "hsj", "clf": clf_name, "eps": "qry",
                              "n": n_sub, "evasion_rate": er,
                              "time_s": round(time.time()-t0, 1)})
        _log(f"  HSJ clf={clf_name}  ER={er:.4f}  ({time.time()-t0:.0f}s)")

    # ── Save generation summary ───────────────────────────────────────
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out = RESULTS_TABLES / "adversarial_generation_summary.csv"
        df_sum.to_csv(out, index=False)
        print(f"\nGeneration summary saved to: {out}")
        print("\n" + df_sum.to_string(index=False))

    print(f"\nTotal time: {time.time()-overall_t0:.1f}s")
    print(f"Adversarial files in: {DATA_ADVERSARIAL}")


if __name__ == "__main__":
    main()
