"""
Experiment 1 — Exploratory Data Analysis
=========================================
Profile every feature that appears in any of the 12 constraints.
Output the observed min/max/mean/percentiles so you can defend each
constraint bound against the data in Section 4 of the paper.

Run:
    python -m experiments.e1_eda
or via run_paper1.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

# allow running from project root or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import RESULTS_TABLES
from grammar.data_loader import load_unsw_normal
from grammar.constraints import CONSTRAINTS


# Features referenced by any constraint
CONSTRAINED_FEATURES = sorted({
    f for spec in CONSTRAINTS.values() for f in spec["features"]
    if f not in ("proto",)  # categorical; profiled separately
})


def profile_feature(s: pd.Series) -> dict:
    """Compute a standard profile for one numeric column."""
    s_num = pd.to_numeric(s, errors="coerce")
    finite = s_num.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) == 0:
        return {
            "n_total": int(len(s)),
            "n_finite": 0,
            "n_nan": int(s_num.isna().sum()),
            "min": np.nan, "p01": np.nan, "p25": np.nan,
            "median": np.nan, "mean": np.nan,
            "p75": np.nan, "p99": np.nan, "max": np.nan, "std": np.nan,
        }
    return {
        "n_total": int(len(s)),
        "n_finite": int(len(finite)),
        "n_nan": int(s_num.isna().sum()),
        "min": float(finite.min()),
        "p01": float(finite.quantile(0.01)),
        "p25": float(finite.quantile(0.25)),
        "median": float(finite.median()),
        "mean": float(finite.mean()),
        "p75": float(finite.quantile(0.75)),
        "p99": float(finite.quantile(0.99)),
        "max": float(finite.max()),
        "std": float(finite.std()),
    }


def main() -> Path:
    print("=" * 70)
    print("E1 — Feature audit on clean (label=0) UNSW-NB15 traffic")
    print("=" * 70)

    normal = load_unsw_normal()
    print(f"\nLoaded {len(normal):,} clean rows, {normal.shape[1]} columns")

    rows = []
    for feat in CONSTRAINED_FEATURES:
        if feat not in normal.columns:
            print(f"  [SKIP] {feat} not present in DataFrame")
            continue
        prof = profile_feature(normal[feat])
        prof = {"feature": feat, **prof}
        rows.append(prof)

    eda = pd.DataFrame(rows)
    out_path = RESULTS_TABLES / "eda_feature_stats.csv"
    eda.to_csv(out_path, index=False)
    print(f"\nSaved feature profile to: {out_path}")

    # print a friendly summary
    print("\n" + "-" * 70)
    print(f"{'feature':<14}{'min':>14}{'p01':>14}{'p99':>14}{'max':>14}")
    print("-" * 70)
    for _, r in eda.iterrows():
        print(f"{r['feature']:<14}{r['min']:>14.3f}{r['p01']:>14.3f}{r['p99']:>14.3f}{r['max']:>14.3f}")
    print("-" * 70)

    # also print proto distribution since several constraints branch on it
    if "proto" in normal.columns:
        print(f"\nproto value counts (top 10):")
        for v, n in normal["proto"].value_counts().head(10).items():
            print(f"  {v:<10} {n:>12,}")

    return out_path


if __name__ == "__main__":
    main()
