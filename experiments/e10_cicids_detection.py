"""
Experiment 10 — CIC-IDS-2017 Grammar Detection + Cross-Dataset Comparison
=========================================================================
Mirrors e6 for CIC-IDS-2017, then produces the cross-dataset comparison
table that is the paper's Table 5 / Figure 8:

    results/tables/cicids_detection_metrics.csv
    results/tables/cross_dataset_comparison.csv  ← the headline table
    results/figures/fig8_cicids_detection.png
    results/figures/fig9_cross_dataset_comparison.png

Run:
    python -m experiments.e10_cicids_detection
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CICIDS_MODELS_DIR, CICIDS_ADV_DIR, RESULTS_TABLES, RESULTS_FIGS
from pipeline.preprocessor import Preprocessor
from pipeline.classifiers import load_model
from pipeline.adversarial import (
    compute_detection_metrics,
    per_constraint_vr,
)
from grammar.constraints_cicids import CICIDS_CONSTRAINTS
from grammar.validator import validate_dataframe as _validate_unsw

plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.labelsize":10,
                     "savefig.dpi":300,"savefig.bbox":"tight"})


def _log(msg): print(f"[e10] {msg}", flush=True)


def _validate_cicids(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 5 CIC-IDS-2017 constraints and return a validation DataFrame."""
    out = pd.DataFrame(index=df.index)
    for cid, spec in CICIDS_CONSTRAINTS.items():
        out[cid] = spec["fn"](df).astype(bool).values
    out["all_valid"] = out.all(axis=1)
    return out


def _reconstruct_for_grammar(X_adv, preprocessor, df_orig):
    """Inverse-scale numeric features (no categoricals in CIC-IDS-2017)."""
    return preprocessor.inverse_numeric(X_adv)


def _evaluate_corpus(path: Path, pre: Preprocessor, clf, df_orig: pd.DataFrame) -> dict:
    stem = path.stem  # adv_fgsm_mlp_eps0p100
    parts = stem.split("_", 3)
    meta = {"attack": parts[1] if len(parts)>1 else "?",
            "clf":    parts[2] if len(parts)>2 else "?",
            "eps_tag":parts[3] if len(parts)>3 else "?"}

    df_adv = pd.read_parquet(path)
    y_true = df_adv["y_true"].values.astype(int)
    feature_cols = pre.feature_names_
    X_adv = df_adv[[c for c in feature_cols if c in df_adv.columns]].values.astype(np.float32)
    n = len(X_adv)

    y_clf = clf.predict(X_adv)
    df_gram = _reconstruct_for_grammar(X_adv, pre, df_orig.iloc[:n])
    gram = _validate_cicids(df_gram)
    grammar_valid = gram["all_valid"].values.astype(bool)

    evading_mask = (y_clf != 1)
    metrics = compute_detection_metrics(y_clf, grammar_valid, attack_label=1)
    pc_vr   = per_constraint_vr(gram, evading_mask, list(CICIDS_CONSTRAINTS.keys()))

    return {**meta, **metrics, "per_constraint_vr": pc_vr}


def _fig8(results, prefix="cicids") -> Path:
    """Stacked bar: caught / grammar adds / missed."""
    labels = [f"{r['attack'].upper()}\n{r['clf']}\n{r['eps_tag']}" for r in results]
    clf_only  = [r["clf_only_tpr"]                         for r in results]
    gram_adds = [r["combined_tpr"] - r["clf_only_tpr"]     for r in results]
    missed    = [1.0 - r["combined_tpr"]                   for r in results]
    x = np.arange(len(results)); w = 0.55
    fig, ax = plt.subplots(figsize=(max(6, len(results)*0.75), 5))
    ax.bar(x, clf_only,  w, color="#2196F3", label="Caught by classifier")
    ax.bar(x, gram_adds, w, bottom=clf_only, color="#4CAF50",
           label="Grammar additionally catches (C2 contribution)")
    ax.bar(x, missed, w,
           bottom=[a+b for a,b in zip(clf_only,gram_adds)],
           color="#F44336", label="Evaded both (ETR)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Fraction of attack flows"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Detection coverage — CIC-IDS-2017")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    out = RESULTS_FIGS / f"fig8_{prefix}_detection.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")
    return out


def _fig9_cross_dataset(unsw: pd.DataFrame, cicids: pd.DataFrame) -> Path:
    """Side-by-side comparison of mean ETR and combined TPR across datasets."""
    metrics = ["clf_only_tpr", "combined_tpr", "etr", "vr_evading"]
    labels  = ["CLF-alone TPR", "Combined TPR", "ETR", "VR on evaders"]

    def agg(df):
        return [df[m].mean() * 100 for m in metrics]

    u_vals = agg(unsw)
    c_vals = agg(cicids)

    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, u_vals, w, color="#1F4E79", label="UNSW-NB15")
    b2 = ax.bar(x + w/2, c_vals, w, color="#C8A951", label="CIC-IDS-2017")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean % across all attack conditions")
    ax.set_title("Cross-dataset grammar-detection comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    for bar, val in list(zip(b1, u_vals)) + list(zip(b2, c_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    out = RESULTS_FIGS / "fig9_cross_dataset_comparison.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")
    return out


def main() -> None:
    print("=" * 70)
    print("E10 — CIC-IDS-2017 detection + cross-dataset comparison")
    print("=" * 70)

    corpora = sorted(CICIDS_ADV_DIR.glob("adv_*.parquet"))
    if not corpora:
        print(f"  No adversarial files in {CICIDS_ADV_DIR}")
        print("  Run python run_pipeline.py --only cicids_adversarials first.")
        return

    _log(f"found {len(corpora)} corpora")
    pre = Preprocessor.load(CICIDS_MODELS_DIR / "preprocessor.pkl")
    orig_path = CICIDS_ADV_DIR / "test_attack_orig.parquet"
    if not orig_path.exists():
        print("  test_attack_orig.parquet missing — re-run e9.")
        return
    df_orig = pd.read_parquet(orig_path)

    clfs = {}
    for name in ("rf", "mlp"):
        try: clfs[name] = load_model(name, models_dir=CICIDS_MODELS_DIR)
        except FileNotFoundError: _log(f"WARNING: {name} not found")

    results = []
    t_overall = time.time()
    for corpus in corpora:
        meta = corpus.stem.split("_", 3)
        clf_name = meta[2] if len(meta) > 2 else "?"
        if clf_name not in clfs:
            _log(f"skipping {corpus.name} — {clf_name} not loaded"); continue
        _log(f"evaluating {corpus.name} ...")
        r = _evaluate_corpus(corpus, pre, clfs[clf_name], df_orig)
        results.append(r)

    if not results:
        print("  No results computed.")
        return

    # ── Save metrics ──────────────────────────────────────────────
    cols = ["attack","clf","eps_tag","n_total","evasion_rate","vr_all",
            "vr_evading","dr","clf_only_tpr","combined_tpr","etr"]
    df_m = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    for c in ["evasion_rate","vr_all","vr_evading","dr","clf_only_tpr","combined_tpr","etr"]:
        df_m[c+"_pct"] = (df_m[c]*100).round(2)
    out = RESULTS_TABLES / "cicids_detection_metrics.csv"
    df_m.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    # ── Print table ───────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Attack':<8}{'CLF':<5}{'Eps':<12}{'ER%':>8}{'VR%':>8}"
          f"{'DR%':>8}{'CLF-TPR%':>10}{'CMB-TPR%':>10}{'ETR%':>8}")
    print("=" * 95)
    for _, r in df_m.iterrows():
        print(f"{r['attack']:<8}{r['clf']:<5}{r['eps_tag']:<12}"
              f"{r['evasion_rate_pct']:>8.2f}{r['vr_all_pct']:>8.2f}"
              f"{r['vr_evading_pct']:>8.2f}{r['clf_only_tpr_pct']:>10.2f}"
              f"{r['combined_tpr_pct']:>10.2f}{r['etr_pct']:>8.2f}")
    print("=" * 95)
    print(f"\n  Mean ER: {df_m['evasion_rate'].mean()*100:.2f}%  "
          f"Mean DR: {df_m['vr_evading'].mean()*100:.2f}%  "
          f"Mean ETR: {df_m['etr'].mean()*100:.2f}%  "
          f"Mean lift: +{(df_m['combined_tpr']-df_m['clf_only_tpr']).mean()*100:.2f}pp")

    # ── Cross-dataset comparison ──────────────────────────────────
    unsw_path = RESULTS_TABLES / "detection_metrics.csv"
    if unsw_path.exists():
        df_unsw = pd.read_csv(unsw_path)
        # Build cross-dataset summary
        def _agg(df, ds):
            return {
                "dataset": ds,
                "mean_er_pct":         df["evasion_rate"].mean()*100,
                "mean_vr_all_pct":     df["vr_all"].mean()*100,
                "mean_vr_evading_pct": df["vr_evading"].mean()*100,
                "mean_dr_pct":         df["dr"].mean()*100,
                "mean_clf_tpr_pct":    df["clf_only_tpr"].mean()*100,
                "mean_combined_tpr_pct": df["combined_tpr"].mean()*100,
                "mean_etr_pct":        df["etr"].mean()*100,
                "tpr_lift_pp":         (df["combined_tpr"]-df["clf_only_tpr"]).mean()*100,
            }
        cross = pd.DataFrame([_agg(df_unsw, "UNSW-NB15"), _agg(df_m, "CIC-IDS-2017")])
        cross_out = RESULTS_TABLES / "cross_dataset_comparison.csv"
        cross.to_csv(cross_out, index=False)
        print(f"\nCross-dataset comparison → {cross_out}")
        print(cross.to_string(index=False))
        _fig9_cross_dataset(df_unsw, df_m)
    else:
        print("\n  UNSW-NB15 detection_metrics.csv not found — "
              "run python run_pipeline.py --only detection first.")

    _fig8(results)
    print(f"\nTotal: {time.time()-t_overall:.1f}s")


if __name__ == "__main__":
    main()
