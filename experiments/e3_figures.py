"""
Experiment 3 — Paper figures
=============================
Reads results/tables/calibration_table.csv (produced by e2_calibration.py)
and writes the three figures referenced in the Paper 1 manuscript.

* Figure 1 — Per-constraint FPR bar chart with 2% threshold line
* Figure 2 — Per-tier FPR comparison
* Figure 3 — Bootstrap CI plot

All saved at 300 DPI as PNG into results/figures/.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering — safe on servers
import matplotlib.pyplot as plt

from config import (
    RESULTS_TABLES,
    RESULTS_FIGS,
    GRAMMAR_FPR_ACTIVE_LIMIT,
    GRAMMAR_FPR_TIER2_LIMIT,
)


# Matplotlib style — keep it boring, journal-ready
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _load_calibration() -> pd.DataFrame:
    path = RESULTS_TABLES / "calibration_table.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"\n  Calibration table not found at {path}\n"
            f"  Run `python -m experiments.e2_calibration` first.\n"
        )
    cal = pd.read_csv(path)
    # Drop the overall-mean summary row
    cal = cal[cal["constraint_id"] != "OVERALL_ACTIVE_MEAN"].copy()
    # Drop constraints that were skipped due to missing columns —
    # they have NaN FPR and cannot be plotted meaningfully.
    skipped = cal["fpr_pct"].isna()
    if skipped.any():
        n_skipped = int(skipped.sum())
        skipped_ids = cal.loc[skipped, "constraint_id"].tolist()
        print(f"  [figures] skipping {n_skipped} N/A constraint(s): {skipped_ids}")
        cal = cal.loc[~skipped].reset_index(drop=True)
    if len(cal) == 0:
        raise RuntimeError(
            "All constraints were skipped (N/A). Cannot generate figures.\n"
            "Your dataset is missing the columns required by every constraint."
        )
    return cal


def figure1_fpr_barchart(cal: pd.DataFrame) -> Path:
    """Horizontal bar chart of per-constraint FPR with 2%/5% threshold lines."""
    cal_sorted = cal.sort_values("fpr_pct", ascending=True)
    fpr_pct = cal_sorted["fpr_pct"].values
    ids = cal_sorted["constraint_id"].values
    tiers = cal_sorted["tier"].values

    fig, ax = plt.subplots(figsize=(8, 5.5))

    colours = []
    for fpr, tier in zip(fpr_pct, tiers):
        if fpr > GRAMMAR_FPR_TIER2_LIMIT * 100:
            colours.append("#C00000")    # red — exceeds 5%
        elif fpr > GRAMMAR_FPR_ACTIVE_LIMIT * 100:
            colours.append("#E69500")    # amber — between 2% and 5%
        else:
            colours.append("#2E7D32")    # green — below 2%

    bars = ax.barh(ids, fpr_pct, color=colours, edgecolor="black", linewidth=0.4)

    ax.axvline(GRAMMAR_FPR_ACTIVE_LIMIT * 100, color="#444", linestyle="--",
               linewidth=1.0, label=f"{GRAMMAR_FPR_ACTIVE_LIMIT*100:.0f}% target")
    ax.axvline(GRAMMAR_FPR_TIER2_LIMIT * 100, color="#C00000", linestyle=":",
               linewidth=1.0, label=f"{GRAMMAR_FPR_TIER2_LIMIT*100:.0f}% Tier-2 cutoff")

    ax.set_xlabel("False Positive Rate on Clean Traffic (%)")
    ax.set_title("Per-constraint FPR  —  UNSW-NB15 Normal Traffic")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    out = RESULTS_FIGS / "fig1_calibration_fpr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def figure2_per_tier_fpr(cal: pd.DataFrame) -> Path:
    """Box-plot style comparison of FPR between Tier 1 and Tier 2."""
    fig, ax = plt.subplots(figsize=(5.5, 4))

    tier1 = cal.loc[cal["tier"] == 1, "fpr_pct"].values
    tier2 = cal.loc[cal["tier"] == 2, "fpr_pct"].values

    positions = [1, 2]
    data = [tier1, tier2]
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=True, medianprops={"color": "black"})
    for patch, c in zip(bp["boxes"], ["#90CAF9", "#FFCC80"]):
        patch.set_facecolor(c)

    # scatter individual points
    for x, d in zip(positions, data):
        if len(d) > 0:
            jitter = (np.random.default_rng(0).random(len(d)) - 0.5) * 0.15
            ax.scatter(np.full(len(d), x) + jitter, d, color="#222",
                       s=18, alpha=0.7, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Tier 1 (RFC-direct)", "Tier 2 (Derived)"])
    ax.set_ylabel("FPR on clean traffic (%)")
    ax.set_title("FPR distribution by constraint tier")
    ax.axhline(GRAMMAR_FPR_ACTIVE_LIMIT * 100, color="#444", linestyle="--",
               linewidth=0.8, label=f"{GRAMMAR_FPR_ACTIVE_LIMIT*100:.0f}% target")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    out = RESULTS_FIGS / "fig2_per_tier_fpr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def figure3_bootstrap_ci(cal: pd.DataFrame) -> Path:
    """Per-constraint FPR with 95% bootstrap CI error bars."""
    cal_sorted = cal.sort_values("fpr_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(cal_sorted))
    fpr = cal_sorted["fpr_pct"].values
    lo = cal_sorted["fpr_ci_lo_pct"].values
    hi = cal_sorted["fpr_ci_hi_pct"].values
    err_lo = np.maximum(fpr - lo, 0.0)
    err_hi = np.maximum(hi - fpr, 0.0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(fpr, y, xerr=[err_lo, err_hi], fmt="o", color="#1F4E79",
                ecolor="#888", capsize=3, markersize=5, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(cal_sorted["constraint_id"].values)
    ax.axvline(GRAMMAR_FPR_ACTIVE_LIMIT * 100, color="#444", linestyle="--",
               linewidth=0.8, label=f"{GRAMMAR_FPR_ACTIVE_LIMIT*100:.0f}% target")
    ax.set_xlabel("FPR on clean traffic (%)  with 95% bootstrap CI")
    ax.set_title("Per-constraint FPR with bootstrap confidence intervals")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    out = RESULTS_FIGS / "fig3_bootstrap_ci.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main() -> list[Path]:
    print("=" * 70)
    print("E3 — Generating Paper 1 figures")
    print("=" * 70)
    cal = _load_calibration()
    paths = [
        figure1_fpr_barchart(cal),
        figure2_per_tier_fpr(cal),
        figure3_bootstrap_ci(cal),
    ]
    print(f"\nAll figures written to: {RESULTS_FIGS}")
    return paths


if __name__ == "__main__":
    main()
