"""
Experiment 9 — CIC-IDS-2017 Adversarial Corpus Generation
===========================================================
Mirrors e5 but for CIC-IDS-2017.  Loads Friday attack flows,
generates adversarials, saves to data/adversarial/cicids/.

Run:
    python -m experiments.e9_cicids_adversarials
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import CICIDS_MODELS_DIR, CICIDS_ADV_DIR, RESULTS_TABLES
from grammar.data_loader_cicids import load_cicids_train_test
from pipeline.preprocessor import Preprocessor
from pipeline.classifiers import load_model
from pipeline.adversarial import (
    ART_AVAILABLE,
    generate_random, generate_fgsm, generate_pgd, generate_hopskipjump,
)

_CICIDS_DROP = ["label", "label_binary"]
_CICIDS_CATS = []

ATTACK_CONFIG = {
    "random": {"epsilons": [0.05, 0.1, 0.3], "classifiers": ["rf", "mlp"]},
    "fgsm":   {"epsilons": [0.05, 0.1, 0.3], "classifiers": ["rf", "mlp"], "n_samples": 1000},
    "pgd":    {"epsilons": [0.1, 0.3],        "classifiers": ["rf", "mlp"], "n_samples": 500,
               "max_iter": 10},
    "hsj":    {"max_eval": 100, "max_iter": 20, "classifiers": ["rf", "mlp"], "n_samples": 200},
}


def _path(attack: str, clf: str, tag: str) -> Path:
    return CICIDS_ADV_DIR / f"adv_{attack}_{clf}_{tag}.parquet"


def _save(X_orig, X_adv, y, pre, path):
    df = pd.DataFrame(X_adv, columns=pre.feature_names_)
    df["y_true"] = y
    df["delta_linf"] = np.abs(X_adv - X_orig).max(axis=1)
    df["delta_l2"]   = np.linalg.norm(X_adv - X_orig, axis=1)
    df.to_parquet(path, index=False)
    print(f"[e9]   saved {len(df):,} → {path.name}")


def main() -> None:
    print("=" * 70)
    print("E9 — CIC-IDS-2017 adversarial corpus generation")
    print("=" * 70)
    if not ART_AVAILABLE:
        print("  NOTE: ART not installed — HopSkipJump will be skipped.")

    overall_t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────
    print("\nLoading Friday test split ...")
    _, test_df = load_cicids_train_test(use_cache=True)

    pre_path = CICIDS_MODELS_DIR / "preprocessor.pkl"
    if not pre_path.exists():
        print(f"  Preprocessor not found: {pre_path}")
        print("  Run python run_pipeline.py --only cicids_classifiers first.")
        return
    pre = Preprocessor.load(pre_path)

    X_test, y_test = pre.transform(test_df)

    # Attack-class flows only
    attack_mask = (y_test == 1)
    X_atk = X_test[attack_mask]
    y_atk = y_test[attack_mask]
    df_atk_orig = test_df[attack_mask.astype(bool)].reset_index(drop=True)

    # Save original attack rows for e10 reconstruction
    df_atk_orig.to_parquet(CICIDS_ADV_DIR / "test_attack_orig.parquet", index=False)
    np.save(CICIDS_ADV_DIR / "test_attack_scaled.npy", X_atk)
    print(f"Attack-class test flows: {len(X_atk):,}")

    # ── Load classifiers ──────────────────────────────────────────
    clfs = {}
    for name in ("rf", "mlp"):
        p = CICIDS_MODELS_DIR / f"{name}.pkl"
        if p.exists():
            clfs[name] = load_model(name, models_dir=CICIDS_MODELS_DIR)
        else:
            print(f"  WARNING: {name}.pkl not found — skipping")

    summary = []

    # ── Random ────────────────────────────────────────────────────
    for clf_name in ATTACK_CONFIG["random"]["classifiers"]:
        if clf_name not in clfs: continue
        clf = clfs[clf_name]
        for eps in ATTACK_CONFIG["random"]["epsilons"]:
            tag = f"eps{eps:.3f}".replace(".", "p")
            p = _path("random", clf_name, tag)
            if p.exists(): continue
            t0 = time.time()
            X_adv = generate_random(X_atk, eps=eps)
            _save(X_atk, X_adv, y_atk, pre, p)
            er = float((clf.predict(X_adv) != 1).mean())
            summary.append({"attack":"random","clf":clf_name,"eps":eps,
                             "n":len(X_adv),"evasion_rate":er,"time_s":round(time.time()-t0,1)})
            print(f"[e9]   Random eps={eps}  clf={clf_name}  ER={er:.4f}")

    # ── FGSM ──────────────────────────────────────────────────────
    cfg = ATTACK_CONFIG["fgsm"]
    n_sub = min(cfg["n_samples"], len(X_atk))
    Xs, ys = X_atk[:n_sub], y_atk[:n_sub]
    for clf_name in cfg["classifiers"]:
        if clf_name not in clfs: continue
        clf = clfs[clf_name]
        for eps in cfg["epsilons"]:
            tag = f"eps{eps:.3f}".replace(".", "p")
            p = _path("fgsm", clf_name, tag)
            if p.exists(): print(f"[e9] fgsm {clf_name} eps={eps} exists — skip"); continue
            t0 = time.time()
            X_adv = generate_fgsm(clf, Xs, eps=eps)
            _save(Xs, X_adv, ys, pre, p)
            er = float((clf.predict(X_adv) != 1).mean())
            summary.append({"attack":"fgsm","clf":clf_name,"eps":eps,
                             "n":n_sub,"evasion_rate":er,"time_s":round(time.time()-t0,1)})
            print(f"[e9]   FGSM eps={eps}  clf={clf_name}  ER={er:.4f}  ({time.time()-t0:.0f}s)")

    # ── PGD ───────────────────────────────────────────────────────
    cfg = ATTACK_CONFIG["pgd"]
    n_sub = min(cfg["n_samples"], len(X_atk))
    Xs, ys = X_atk[:n_sub], y_atk[:n_sub]
    for clf_name in cfg["classifiers"]:
        if clf_name not in clfs: continue
        clf = clfs[clf_name]
        for eps in cfg["epsilons"]:
            tag = f"eps{eps:.3f}".replace(".", "p")
            p = _path("pgd", clf_name, tag)
            if p.exists(): print(f"[e9] pgd {clf_name} eps={eps} exists — skip"); continue
            t0 = time.time()
            X_adv = generate_pgd(clf, Xs, eps=eps, max_iter=cfg["max_iter"])
            _save(Xs, X_adv, ys, pre, p)
            er = float((clf.predict(X_adv) != 1).mean())
            summary.append({"attack":"pgd","clf":clf_name,"eps":eps,
                             "n":n_sub,"evasion_rate":er,"time_s":round(time.time()-t0,1)})
            print(f"[e9]   PGD eps={eps}  clf={clf_name}  ER={er:.4f}  ({time.time()-t0:.0f}s)")

    # ── HopSkipJump ───────────────────────────────────────────────
    if ART_AVAILABLE:
        cfg = ATTACK_CONFIG["hsj"]
        n_sub = min(cfg["n_samples"], len(X_atk))
        Xs, ys = X_atk[:n_sub], y_atk[:n_sub]
        for clf_name in cfg["classifiers"]:
            if clf_name not in clfs: continue
            clf = clfs[clf_name]
            p = _path("hsj", clf_name, f"n{n_sub}")
            if p.exists(): print(f"[e9] hsj {clf_name} exists — skip"); continue
            t0 = time.time()
            X_adv = generate_hopskipjump(clf, Xs, max_eval=cfg["max_eval"],
                                         max_iter=cfg["max_iter"])
            _save(Xs, X_adv, ys, pre, p)
            er = float((clf.predict(X_adv) != 1).mean())
            summary.append({"attack":"hsj","clf":clf_name,"eps":"qry",
                             "n":n_sub,"evasion_rate":er,"time_s":round(time.time()-t0,1)})
            print(f"[e9]   HSJ clf={clf_name}  ER={er:.4f}  ({time.time()-t0:.0f}s)")

    if summary:
        df_sum = pd.DataFrame(summary)
        out = RESULTS_TABLES / "cicids_adversarial_summary.csv"
        df_sum.to_csv(out, index=False)
        print(f"\nSummary → {out}")
        print(df_sum.to_string(index=False))

    print(f"\nTotal: {time.time()-overall_t0:.1f}s  |  files in: {CICIDS_ADV_DIR}")


if __name__ == "__main__":
    main()
