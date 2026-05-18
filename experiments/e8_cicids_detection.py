"""
Experiment 8 — CIC-IDS-2017 classifier training + detection evaluation
=======================================================================
Trains RF + MLP on CIC-IDS-2017, generates FGSM/PGD adversarials,
applies grammar detection, and produces the cross-dataset detection
metrics table for the manuscript.

This is the evidence that the C2 claim generalises beyond UNSW-NB15.

Run:
    python -m experiments.e8_cicids_detection
or via:
    python run_pipeline.py --only cicids_detection
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (
    CICIDS_MODELS_DIR, CICIDS_ADV_DIR, RESULTS_TABLES,
    RF_PARAMS, MLP_PARAMS,
)
from grammar.data_loader_cicids import load_cicids_train_test
from grammar.constraints_cicids import CICIDS_CONSTRAINTS
from grammar.validator import validate_dataframe as _validate_df
from pipeline.preprocessor import Preprocessor
from pipeline.classifiers import (
    train_rf, train_mlp, save_model, load_model,
    evaluate_model, metrics_table,
)
from pipeline.adversarial import (
    generate_fgsm, generate_pgd,
    reconstruct_for_grammar, compute_detection_metrics,
)


# CIC-IDS-2017-aware validator that uses the CIC-IDS constraint set
def _validate_cicids(df: pd.DataFrame) -> pd.DataFrame:
    from grammar.validator import _all_valid
    import pandas as pd
    out = pd.DataFrame(index=df.index)
    for cid, spec in CICIDS_CONSTRAINTS.items():
        try:
            out[cid] = spec["fn"](df).astype(bool).values
        except Exception:
            out[cid] = True
    out["all_valid"] = out.all(axis=1)
    return out


def _log(msg: str) -> None:
    print(f"[e8] {msg}", flush=True)


# Drop columns the grammar validator doesn't care about for CIC-IDS-2017
CICIDS_DROP = [
    "flow_id", "source_ip", "source_port",
    "destination_ip", "destination_port",
    "timestamp", "label_str",
]


def main() -> None:
    print("=" * 70)
    print("E8 — CIC-IDS-2017 classifier training + C2 detection")
    print("=" * 70)

    overall_t0 = time.time()

    # ── Load data ────────────────────────────────────────────────────
    _log("loading CIC-IDS-2017 train/test split ...")
    train_df, test_df = load_cicids_train_test(use_cache=True)
    print(f"\n  train={train_df.shape}  test={test_df.shape}")
    print(f"  train normal={int((train_df['label']==0).sum()):,}  "
          f"attack={int((train_df['label']==1).sum()):,}")
    print(f"  test  normal={int((test_df['label']==0).sum()):,}   "
          f"attack={int((test_df['label']==1).sum()):,}")

    # ── Fit preprocessor ─────────────────────────────────────────────
    pre_path = CICIDS_MODELS_DIR / "preprocessor.pkl"
    if pre_path.exists():
        _log("loading cached CIC-IDS-2017 preprocessor ...")
        pre = Preprocessor.load(pre_path)
        X_train, y_train = pre.transform(train_df)
        X_test, y_test = pre.transform(test_df)
    else:
        _log("fitting preprocessor on CIC-IDS-2017 train ...")
        pre = Preprocessor()
        X_train, y_train = pre.fit_transform(train_df)
        X_test, y_test = pre.transform(test_df)
        pre.save(pre_path)

    print(f"\n  X_train: {X_train.shape}   X_test: {X_test.shape}")

    # ── Train classifiers ─────────────────────────────────────────────
    classifiers = {}
    for name, train_fn in (("rf", train_rf), ("mlp", train_mlp)):
        model_path = CICIDS_MODELS_DIR / f"{name}.pkl"
        if model_path.exists():
            _log(f"loading cached {name} ...")
            classifiers[name] = load_model(name, models_dir=CICIDS_MODELS_DIR)
        else:
            _log(f"training {name} ...")
            clf = train_fn(X_train, y_train)
            save_model(clf, name, models_dir=CICIDS_MODELS_DIR)
            classifiers[name] = clf

    # ── Baseline evaluation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Classifier baseline on CIC-IDS-2017 test set")
    print("=" * 70)
    baseline_rows = []
    for name, clf in classifiers.items():
        m = evaluate_model(clf, X_test, y_test, name=f"CICIDS_{name.upper()}")
        baseline_rows.append(m)
        print(f"  {name.upper():6s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
              f"auc={m['roc_auc']:.4f}  tpr={m['tpr']:.4f}  fpr={m['fpr']:.4f}")

    baseline_path = RESULTS_TABLES / "classifier_baseline_cicids.csv"
    metrics_table(baseline_rows).to_csv(baseline_path, index=False)
    _log(f"saved baseline → {baseline_path}")

    # ── Adversarial generation (FGSM + PGD against MLP) ──────────────
    attack_mask = (y_test == 1)
    X_attack = X_test[attack_mask]
    y_attack = y_test[attack_mask]
    # Keep original test rows for grammar reconstruction
    test_attack_df = test_df[attack_mask.astype(bool)].reset_index(drop=True)
    _log(f"attack-class test flows: {len(X_attack):,}")

    ATTACKS = [
        ("fgsm", "mlp", 0.1),
        ("fgsm", "mlp", 0.3),
        ("pgd",  "mlp", 0.1),
        ("pgd",  "mlp", 0.3),
    ]
    N_ADV = min(1000, len(X_attack))
    X_sub = X_attack[:N_ADV]
    y_sub = y_attack[:N_ADV]
    test_sub_df = test_attack_df.iloc[:N_ADV].reset_index(drop=True)

    detection_rows = []
    for atk, clf_name, eps in ATTACKS:
        clf = classifiers.get(clf_name)
        if clf is None:
            continue
        tag = f"{atk}_{clf_name}_eps{eps:.2f}".replace(".", "p")
        adv_path = CICIDS_ADV_DIR / f"{tag}.parquet"
        if not adv_path.exists():
            _log(f"generating {atk.upper()} eps={eps} against {clf_name.upper()} ...")
            t0 = time.time()
            if atk == "fgsm":
                X_adv = generate_fgsm(clf, X_sub, eps=eps)
            else:
                X_adv = generate_pgd(clf, X_sub, eps=eps, max_iter=10)
            # Save
            df_adv = pd.DataFrame(X_adv, columns=pre.feature_names_)
            df_adv["y_true"] = y_sub
            df_adv.to_parquet(adv_path, index=False)
            _log(f"  saved → {adv_path.name}  ({time.time()-t0:.1f}s)")
        else:
            df_adv = pd.read_parquet(adv_path)
            X_adv = df_adv[[c for c in pre.feature_names_ if c in df_adv.columns]].values.astype(np.float32)

        # ── Grammar detection ────────────────────────────────────────
        df_gram = reconstruct_for_grammar(X_adv, pre, test_sub_df)
        gram_result = _validate_cicids(df_gram)
        grammar_valid = gram_result["all_valid"].values.astype(bool)

        y_clf_pred = clf.predict(X_adv)
        m = compute_detection_metrics(y_clf_pred, grammar_valid, attack_label=1)

        detection_rows.append({
            "dataset": "CIC-IDS-2017",
            "attack": atk.upper(),
            "clf": clf_name.upper(),
            "eps": eps,
            "n_total": m["n_total"],
            "evasion_rate_pct": round(m["evasion_rate"] * 100, 2),
            "vr_all_pct": round(m["vr_all"] * 100, 2),
            "vr_evading_pct": round(m["vr_evading"] * 100, 2),
            "clf_only_tpr_pct": round(m["clf_only_tpr"] * 100, 2),
            "combined_tpr_pct": round(m["combined_tpr"] * 100, 2),
            "etr_pct": round(m["etr"] * 100, 2),
        })
        _log(f"  {atk.upper()} eps={eps}  ER={m['evasion_rate']:.3f}  "
             f"VR={m['vr_evading']:.3f}  ETR={m['etr']:.3f}")

    # ── Save and print ────────────────────────────────────────────────
    if detection_rows:
        df_det = pd.DataFrame(detection_rows)
        det_path = RESULTS_TABLES / "detection_metrics_cicids.csv"
        df_det.to_csv(det_path, index=False)

        print("\n" + "=" * 90)
        print(f"{'Attack':<8}{'CLF':<6}{'eps':>6}{'ER%':>8}{'VR%':>8}"
              f"{'DR%':>8}{'CLF-TPR%':>10}{'CMB-TPR%':>10}{'ETR%':>8}")
        print("=" * 90)
        for _, r in df_det.iterrows():
            dr = r["vr_evading_pct"]
            print(f"{r['attack']:<8}{r['clf']:<6}{r['eps']:>6.2f}"
                  f"{r['evasion_rate_pct']:>8.2f}{r['vr_all_pct']:>8.2f}"
                  f"{dr:>8.2f}{r['clf_only_tpr_pct']:>10.2f}"
                  f"{r['combined_tpr_pct']:>10.2f}{r['etr_pct']:>8.2f}")
        print("=" * 90)
        print(f"\nSaved → {det_path}")

    print(f"\nTotal time: {time.time()-overall_t0:.1f}s")


if __name__ == "__main__":
    main()
