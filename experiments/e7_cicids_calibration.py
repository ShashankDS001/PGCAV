"""
Experiment 7 — CIC-IDS-2017 Grammar Calibration
================================================
Mirrors e2_calibration.py but for CIC-IDS-2017 using the 5 transferable
constraints in grammar/constraints_cicids.py.

Outputs:
    results/tables/cicids_calibration_table.csv
    results/figures/fig7_cicids_fpr.png

Run:
    python -m experiments.e7_cicids_calibration
or:
    python run_pipeline.py --only cicids_calibration
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

from config import (
    RESULTS_TABLES, RESULTS_FIGS,
    GRAMMAR_FPR_ACTIVE_LIMIT, GRAMMAR_FPR_TIER2_LIMIT,
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SAMPLE_SIZE, RANDOM_SEED,
)
from grammar.data_loader_cicids import load_cicids_normal
from grammar.constraints_cicids import CICIDS_CONSTRAINTS


def _bootstrap_fpr_ci(valid: np.ndarray, n_res: int = BOOTSTRAP_RESAMPLES,
                      sample_size: int = BOOTSTRAP_SAMPLE_SIZE) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(valid)
    ss = min(sample_size, n)
    fprs = np.array([
        1.0 - valid[rng.integers(0, n, ss)].mean()
        for _ in range(n_res)
    ])
    return float(np.percentile(fprs, 2.5)), float(np.percentile(fprs, 97.5))


def calibrate_cicids(df_normal: pd.DataFrame, bootstrap: bool = True) -> pd.DataFrame:
    n = len(df_normal)
    print(f"[e7] calibrating {len(CICIDS_CONSTRAINTS)} constraints on {n:,} rows")
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for cid, spec in CICIDS_CONSTRAINTS.items():
        missing = [f for f in spec["features"] if f not in df_normal.columns]
        if missing:
            print(f"[e7]   {cid}: N/A — missing {missing}")
            rows.append({
                "constraint_id": cid, "tier": spec["tier"],
                "unsw_equivalent": spec["unsw_equivalent"],
                "description": spec["description"],
                "missing_features": ",".join(missing),
                "n_rows": n, "n_violations": np.nan,
                "fpr": np.nan, "fpr_pct": np.nan,
                "fpr_ci_lo_pct": np.nan, "fpr_ci_hi_pct": np.nan,
                "status": f"N/A — column(s) missing: {','.join(missing)}",
            })
            continue

        print(f"[e7]   evaluating {cid} ...")
        valid = spec["fn"](df_normal).astype(bool).values
        n_viol = int((~valid).sum())
        fpr = n_viol / n

        if bootstrap and n > 1:
            lo, hi = _bootstrap_fpr_ci(valid)
        else:
            lo, hi = fpr, fpr

        if fpr <= GRAMMAR_FPR_ACTIVE_LIMIT:
            status = "Active"
        elif fpr <= GRAMMAR_FPR_TIER2_LIMIT:
            status = "Active (above 2% target)"
        else:
            status = "Demoted to Informational"

        rows.append({
            "constraint_id": cid, "tier": spec["tier"],
            "unsw_equivalent": spec["unsw_equivalent"],
            "description": spec["description"],
            "missing_features": "",
            "n_rows": n, "n_violations": n_viol,
            "fpr": fpr, "fpr_pct": fpr * 100.0,
            "fpr_ci_lo_pct": lo * 100.0,
            "fpr_ci_hi_pct": hi * 100.0,
            "status": status,
        })

    cal = pd.DataFrame(rows).sort_values("fpr", ascending=False, na_position="last")
    return cal.reset_index(drop=True)


def _figure7(cal: pd.DataFrame) -> Path:
    df = cal.dropna(subset=["fpr_pct"]).sort_values("fpr_pct", ascending=True)
    colours = ["#C00000" if r["fpr_pct"] > GRAMMAR_FPR_TIER2_LIMIT * 100
               else "#E69500" if r["fpr_pct"] > GRAMMAR_FPR_ACTIVE_LIMIT * 100
               else "#2E7D32" for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["constraint_id"], df["fpr_pct"], color=colours,
            edgecolor="black", linewidth=0.4)
    ax.axvline(GRAMMAR_FPR_ACTIVE_LIMIT * 100, color="#444", linestyle="--",
               linewidth=0.9, label="2% target")
    ax.set_xlabel("FPR on CIC-IDS-2017 benign traffic (%)")
    ax.set_title("CIC-IDS-2017 grammar calibration (5 applicable constraints)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    out = RESULTS_FIGS / "fig7_cicids_fpr.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main() -> Path:
    print("=" * 70)
    print("E7 — CIC-IDS-2017 grammar calibration")
    print("=" * 70)

    t0 = time.time()
    normal = load_cicids_normal()
    print(f"Loaded {len(normal):,} benign CIC-IDS-2017 rows ({time.time()-t0:.1f}s)")

    cal = calibrate_cicids(normal, bootstrap=True)

    out = RESULTS_TABLES / "cicids_calibration_table.csv"
    cal.to_csv(out, index=False)
    print(f"\nSaved to: {out}")

    # Print table
    print("\n" + "=" * 95)
    print(f"{'ID':<28}{'UNSW':<6}{'FPR %':>10}  {'95% CI':>18}  {'Status'}")
    print("=" * 95)
    for _, r in cal.iterrows():
        if pd.isna(r["fpr_pct"]):
            print(f"{r['constraint_id']:<28}{r['unsw_equivalent']:<6}{'—':>10}  {'':>18}  {r['status']}")
        else:
            ci = f"[{r['fpr_ci_lo_pct']:>5.3f}, {r['fpr_ci_hi_pct']:>5.3f}]"
            print(f"{r['constraint_id']:<28}{r['unsw_equivalent']:<6}"
                  f"{r['fpr_pct']:>10.4f}  {ci:>18}  {r['status']}")
    print("=" * 95)

    # Gate
    evaluated = cal.dropna(subset=["fpr"])
    active = evaluated[evaluated["status"].str.startswith("Active")]
    n_eval = len(evaluated); n_skip = len(cal) - n_eval
    mean_fpr = float(active["fpr"].mean()) if len(active) else 0.0
    max_fpr  = float(np.nanmax(active["fpr"].values)) if len(active) else 0.0

    print(f"\n  Constraints: {n_eval} evaluated, {n_skip} N/A")
    print(f"  Mean Active FPR : {mean_fpr*100:.4f}%  "
          f"→ {'PASS' if mean_fpr <= GRAMMAR_FPR_ACTIVE_LIMIT else 'FAIL'}")
    print(f"  Max Active FPR  : {max_fpr*100:.4f}%  "
          f"→ {'PASS' if max_fpr <= GRAMMAR_FPR_TIER2_LIMIT else 'FAIL'}")

    _figure7(cal)
    return out


if __name__ == "__main__":
    main()
