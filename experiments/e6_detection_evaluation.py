"""
Experiment 6 — Grammar-as-Detector evaluation  (Phase 2C, C2 framing)
======================================================================
Loads each adversarial corpus from data/adversarial/, inverse-transforms
to the original feature space, applies the grammar validator, and computes
the full C2 detection metric suite:

    ER   — Evasion Rate (fraction that fool the ML classifier)
    VR   — Violation Rate (fraction that violate >= 1 grammar constraint)
    DR   — Detection Rate (VR on the EVADING subset)
    TPR_combined — 1 - ER * (1 - DR)  [combined classifier + grammar]
    ETR  — Effective Threat Rate  [ER * (1 - DR)]

Generates three paper figures:
    fig4_detection_breakdown.png — stacked bar: caught/missed per attack
    fig5_per_constraint_vr.png  — which constraints fire most on adversarials
    fig6_etr_scatter.png        — ETR vs ER scatter (grammar effect)

Run:
    python -m experiments.e6_detection_evaluation
or via:
    python run_pipeline.py --only detection
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

from config import DATA_ADVERSARIAL, MODELS_DIR, RESULTS_TABLES, RESULTS_FIGS
from pipeline.preprocessor import Preprocessor
from pipeline.classifiers import load_model
from pipeline.adversarial import (
    reconstruct_for_grammar,
    compute_detection_metrics,
    per_constraint_vr,
)
from grammar.constraints import CONSTRAINTS
from grammar.validator import validate_dataframe

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _log(msg: str) -> None:
    print(f"[e6] {msg}", flush=True)


def _list_corpora() -> list[Path]:
    """Return all adversarial Parquet files sorted by name."""
    return sorted(DATA_ADVERSARIAL.glob("adv_*.parquet"))


def _parse_corpus_name(path: Path) -> dict:
    """Parse attack, clf, eps from filename like adv_fgsm_mlp_eps0p100.parquet"""
    stem = path.stem  # e.g. adv_fgsm_mlp_eps0p100
    parts = stem.split("_", maxsplit=3)  # ['adv', 'fgsm', 'mlp', 'eps0p100']
    return {
        "attack": parts[1] if len(parts) > 1 else "?",
        "clf":    parts[2] if len(parts) > 2 else "?",
        "eps_tag": parts[3] if len(parts) > 3 else "?",
        "label": f"{parts[1].upper()}/{parts[2].upper()}  {parts[3] if len(parts)>3 else ''}",
    }


def evaluate_corpus(
    corpus_path: Path,
    preprocessor: Preprocessor,
    clf,
    df_orig: pd.DataFrame,
) -> dict:
    """
    For one adversarial corpus:
      1. Load scaled adversarials
      2. Run classifier to measure evasion
      3. Reconstruct in original feature space
      4. Run grammar validator
      5. Compute detection metrics
    """
    meta = _parse_corpus_name(corpus_path)

    df_adv = pd.read_parquet(corpus_path)
    y_true = df_adv["y_true"].values.astype(int)
    feature_cols = preprocessor.feature_names_
    X_adv = df_adv[[c for c in feature_cols if c in df_adv.columns]].values.astype(np.float32)

    # Align df_orig to same length
    n = len(X_adv)
    df_orig_n = df_orig.iloc[:n].reset_index(drop=True)

    # 1. Classifier evasion
    y_clf_pred = clf.predict(X_adv)

    # 2. Reconstruct for grammar
    df_gram = reconstruct_for_grammar(X_adv, preprocessor, df_orig_n)

    # 3. Grammar validation
    gram_result = validate_dataframe(df_gram)
    grammar_valid = gram_result["all_valid"].values.astype(bool)

    # 4. Detection metrics
    evading_mask = (y_clf_pred != 1)
    metrics = compute_detection_metrics(y_clf_pred, grammar_valid, attack_label=1)

    # 5. Per-constraint VR
    c_ids = list(CONSTRAINTS.keys())
    pc_vr = per_constraint_vr(gram_result, evading_mask, c_ids)

    return {
        **meta,
        **metrics,
        "per_constraint_vr": pc_vr,
    }


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def _short_label(row: dict) -> str:
    atk = row["attack"].upper()
    eps = row["eps_tag"]
    return f"{atk}\n{eps}"


def figure4_detection_breakdown(results: list[dict]) -> Path:
    """Stacked bar: for each attack, fraction caught by classifier /
    additionally caught by grammar / evading both."""
    labels = [_short_label(r) for r in results]
    clf_only  = [r["clf_only_tpr"]                          for r in results]
    gram_adds = [r["combined_tpr"] - r["clf_only_tpr"]      for r in results]
    missed    = [1.0 - r["combined_tpr"]                    for r in results]

    x = np.arange(len(results))
    w = 0.55
    fig, ax = plt.subplots(figsize=(max(6, len(results)*0.7), 5))
    b1 = ax.bar(x, clf_only,  w, color="#2196F3", label="Caught by classifier")
    b2 = ax.bar(x, gram_adds, w, bottom=clf_only, color="#4CAF50",
                label="Additionally caught by grammar (C2 contribution)")
    b3 = ax.bar(x, missed,    w,
                bottom=[a+b for a,b in zip(clf_only, gram_adds)],
                color="#F44336", label="Evaded both layers (ETR)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of attack flows"); ax.set_ylim(0, 1.0)
    ax.set_title("Detection coverage: classifier vs classifier + grammar")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    out = RESULTS_FIGS / "fig4_detection_breakdown.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")
    return out


def figure5_per_constraint_vr(results: list[dict]) -> Path:
    """Heatmap of per-constraint VR across attack conditions."""
    # Aggregate mean VR_evading per constraint across all conditions
    constraint_ids = list(CONSTRAINTS.keys())
    agg = {cid: [] for cid in constraint_ids}
    for r in results:
        pc = r["per_constraint_vr"].set_index("constraint_id")
        for cid in constraint_ids:
            if cid in pc.index:
                agg[cid].append(pc.loc[cid, "vr_evading"])

    cids_sorted = sorted(agg, key=lambda c: -np.mean(agg[c]) if agg[c] else 0)
    mean_vrs = [np.mean(agg[c]) * 100 if agg[c] else 0.0 for c in cids_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["#C00000" if v > 20 else "#E69500" if v > 5 else "#2E7D32"
               for v in mean_vrs]
    ax.barh(cids_sorted, mean_vrs, color=colours, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean VR on evading adversarials (%) across all attack conditions")
    ax.set_title("Per-constraint detection contribution")
    ax.grid(axis="x", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    out = RESULTS_FIGS / "fig5_per_constraint_vr.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")
    return out


def figure6_etr_scatter(results: list[dict]) -> Path:
    """ER vs ETR scatter — grammar's effect on effective threat."""
    ers  = [r["evasion_rate"] for r in results]
    etrs = [r["etr"]          for r in results]
    labs = [_short_label(r)   for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ers, etrs, color="#1F4E79", s=60, zorder=3)
    for x, y, l in zip(ers, etrs, labs):
        ax.annotate(l, (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    lim = max(max(ers), max(etrs)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="No grammar (ETR=ER)")
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.05, color="red")
    ax.set_xlabel("Evasion Rate (ER) — classifier alone")
    ax.set_ylabel("Effective Threat Rate (ETR) — after grammar detection")
    ax.set_title("Grammar reduces effective adversarial threat")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    out = RESULTS_FIGS / "fig6_etr_scatter.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("E6 — Grammar-as-Detector evaluation")
    print("=" * 70)

    corpora = _list_corpora()
    if not corpora:
        print(f"\n  No adversarial files found in {DATA_ADVERSARIAL}/")
        print("  Run python -m experiments.e5_generate_adversarials first.")
        return

    _log(f"found {len(corpora)} adversarial corpora")

    # Load shared artefacts
    preprocessor = Preprocessor.load(MODELS_DIR / "preprocessor.pkl")
    orig_path = DATA_ADVERSARIAL / "test_attack_orig.parquet"
    if not orig_path.exists():
        print(f"\n  test_attack_orig.parquet not found — re-run e5 to regenerate.")
        return
    df_orig = pd.read_parquet(orig_path)

    classifiers = {}
    for name in ("rf", "mlp"):
        try:
            classifiers[name] = load_model(name)
        except FileNotFoundError:
            _log(f"WARNING: {name}.pkl not found")

    results = []
    t_overall = time.time()
    for corpus in corpora:
        meta = _parse_corpus_name(corpus)
        clf_name = meta["clf"]
        if clf_name not in classifiers:
            _log(f"skipping {corpus.name} — {clf_name} classifier not loaded")
            continue
        _log(f"evaluating {corpus.name} ...")
        t0 = time.time()
        r = evaluate_corpus(corpus, preprocessor, classifiers[clf_name], df_orig)
        r["eval_seconds"] = round(time.time() - t0, 1)
        results.append(r)

    if not results:
        print("  No results computed. Check that classifiers match corpus clf tags.")
        return

    # ── Save metrics table ───────────────────────────────────────────
    cols = ["attack", "clf", "eps_tag", "n_total",
            "evasion_rate", "vr_all", "vr_evading", "dr",
            "clf_only_tpr", "combined_tpr", "etr"]
    df_metrics = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    # Format percentage columns
    for c in ["evasion_rate","vr_all","vr_evading","dr","clf_only_tpr","combined_tpr","etr"]:
        df_metrics[c+"_pct"] = (df_metrics[c] * 100).round(2)
    out_path = RESULTS_TABLES / "detection_metrics.csv"
    df_metrics.to_csv(out_path, index=False)
    print(f"\nSaved detection metrics to: {out_path}")

    # ── Pretty-print ─────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'Attack':<8}{'CLF':<5}{'Eps':<12}{'ER%':>8}{'VR%':>8}"
          f"{'DR%':>8}{'CLF-TPR%':>10}{'CMB-TPR%':>10}{'ETR%':>8}")
    print("=" * 100)
    for _, r in df_metrics.iterrows():
        print(f"{r['attack']:<8}{r['clf']:<5}{r['eps_tag']:<12}"
              f"{r['evasion_rate_pct']:>8.2f}{r['vr_all_pct']:>8.2f}"
              f"{r['vr_evading_pct']:>8.2f}{r['clf_only_tpr_pct']:>10.2f}"
              f"{r['combined_tpr_pct']:>10.2f}{r['etr_pct']:>8.2f}")
    print("=" * 100)

    print(f"\nKey result for manuscript:")
    mean_dr = df_metrics["dr"].mean() * 100
    mean_etr = df_metrics["etr"].mean() * 100
    mean_er  = df_metrics["evasion_rate"].mean() * 100
    lift     = df_metrics["combined_tpr"].mean() * 100 - df_metrics["clf_only_tpr"].mean() * 100
    print(f"  Mean Evasion Rate (ER):              {mean_er:.2f}%")
    print(f"  Mean Detection Rate on evaders (DR): {mean_dr:.2f}%")
    print(f"  Mean Effective Threat Rate (ETR):    {mean_etr:.2f}%")
    print(f"  Grammar TPR lift over clf-alone:     +{lift:.2f}pp")

    # ── Figures ──────────────────────────────────────────────────────
    print(f"\nGenerating figures ...")
    figure4_detection_breakdown(results)
    figure5_per_constraint_vr(results)
    if len(results) >= 2:
        figure6_etr_scatter(results)

    print(f"\nTotal time: {time.time()-t_overall:.1f}s")


if __name__ == "__main__":
    main()
