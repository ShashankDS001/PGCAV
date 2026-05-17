"""
PGCAV Grammar Validator
=======================
Two entry points:

* `validate_dataframe(df)` runs all 12 constraints on a DataFrame and
  returns a long-form table of (constraint_id, row_index, violated).
  This is what you call at inference time for the detection layer.

* `calibrate_constraints(df)` measures the false-positive rate of every
  constraint on a clean-traffic DataFrame, with bootstrap 95% CIs.
  This produces the calibration table that becomes Table 2 in the paper.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SAMPLE_SIZE,
    GRAMMAR_FPR_ACTIVE_LIMIT,
    GRAMMAR_FPR_TIER2_LIMIT,
    RANDOM_SEED,
    VERBOSE,
)
from .constraints import CONSTRAINTS


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[validator] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Per-row validation (single-flow inference)
# ──────────────────────────────────────────────────────────────────────

def validate_row(row: dict | pd.Series) -> dict:
    """Validate a single flow record.

    Parameters
    ----------
    row : dict or pd.Series with lowercase UNSW-NB15 feature keys.

    Returns
    -------
    {"valid": bool, "violated": [constraint_id, ...]}
    """
    df = pd.DataFrame([dict(row)])
    violations = []
    for cid, spec in CONSTRAINTS.items():
        result = spec["fn"](df)
        if not bool(result.iloc[0]):
            violations.append(cid)
    return {"valid": len(violations) == 0, "violated": violations}


def validate_dataframe(
    df: pd.DataFrame,
    constraints: Optional[dict] = None,
) -> pd.DataFrame:
    """Apply all constraints to a DataFrame.

    Returns a DataFrame of the same length with one Boolean column per
    constraint (True = valid, False = violation) plus an `all_valid` column.
    """
    if constraints is None:
        constraints = CONSTRAINTS

    out = pd.DataFrame(index=df.index)
    for cid, spec in constraints.items():
        out[cid] = spec["fn"](df).astype(bool).values
    out["all_valid"] = out.all(axis=1)
    return out


# ──────────────────────────────────────────────────────────────────────
# Calibration (the main Paper 1 experiment)
# ──────────────────────────────────────────────────────────────────────

def _bootstrap_fpr_ci(
    valid_series: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    sample_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Compute a 95% bootstrap CI on the FPR (= 1 - mean(valid))."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n = len(valid_series)
    if sample_size is None or sample_size > n:
        sample_size = n
    fprs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=sample_size)
        fprs[i] = 1.0 - valid_series[idx].mean()
    lo, hi = np.percentile(fprs, [2.5, 97.5])
    return float(lo), float(hi)


def _classify_status(fpr: float, tier: int) -> str:
    """Apply the Doc2 §3.3 go/no-go gate."""
    if fpr <= GRAMMAR_FPR_ACTIVE_LIMIT:
        return "Active"
    if fpr <= GRAMMAR_FPR_TIER2_LIMIT:
        return "Active (above 2% target)"
    return "Demoted to Informational"


def calibrate_constraints(
    df_normal: pd.DataFrame,
    bootstrap: bool = True,
) -> pd.DataFrame:
    """Run all 12 constraints against clean normal traffic.

    Parameters
    ----------
    df_normal : DataFrame of clean (label=0) UNSW-NB15 rows.
    bootstrap : whether to compute bootstrap 95% CIs.

    Returns
    -------
    DataFrame with columns:
        constraint_id, tier, rfc, description, required_features, missing_features,
        n_rows, n_violations, fpr, fpr_ci_lo, fpr_ci_hi, status
    sorted by FPR descending, with constraints whose features are missing
    explicitly marked as N/A — column missing instead of silently passing.
    """
    n = len(df_normal)
    _log(f"calibrating {len(CONSTRAINTS)} constraints on {n:,} clean rows")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for cid, spec in CONSTRAINTS.items():
        required = spec["features"]
        missing = [f for f in required if f not in df_normal.columns]

        if missing:
            # The constraint cannot be evaluated. Do NOT silently pass.
            _log(f"  {cid}: SKIPPED — missing column(s) {missing}")
            rows.append({
                "constraint_id": cid,
                "tier": spec["tier"],
                "rfc": spec["rfc"],
                "description": spec["description"],
                "required_features": ",".join(required),
                "missing_features": ",".join(missing),
                "n_rows": n,
                "n_violations": np.nan,
                "fpr": np.nan,
                "fpr_pct": np.nan,
                "fpr_ci_lo_pct": np.nan,
                "fpr_ci_hi_pct": np.nan,
                "status": f"N/A — column(s) missing: {','.join(missing)}",
            })
            continue

        _log(f"  evaluating {cid} ...")
        valid = spec["fn"](df_normal).astype(bool).values
        n_viol = int((~valid).sum())
        fpr = float(n_viol / n) if n else 0.0

        if bootstrap and n > 1:
            lo, hi = _bootstrap_fpr_ci(
                valid_series=valid,
                sample_size=min(BOOTSTRAP_SAMPLE_SIZE, n),
                rng=rng,
            )
        else:
            lo, hi = fpr, fpr

        rows.append({
            "constraint_id": cid,
            "tier": spec["tier"],
            "rfc": spec["rfc"],
            "description": spec["description"],
            "required_features": ",".join(required),
            "missing_features": "",
            "n_rows": n,
            "n_violations": n_viol,
            "fpr": fpr,
            "fpr_pct": fpr * 100.0,
            "fpr_ci_lo_pct": lo * 100.0,
            "fpr_ci_hi_pct": hi * 100.0,
            "status": _classify_status(fpr, spec["tier"]),
        })

    out = pd.DataFrame(rows).sort_values(
        "fpr", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # Summary row — only over constraints that were ACTUALLY evaluated
    evaluated_mask = out["status"].str.startswith("Active")
    n_evaluated = int(evaluated_mask.sum())
    n_skipped = int(out["status"].str.startswith("N/A").sum())
    mean_fpr_active = (
        float(out.loc[evaluated_mask, "fpr"].mean()) if evaluated_mask.any() else 0.0
    )
    target_ok = mean_fpr_active <= GRAMMAR_FPR_ACTIVE_LIMIT
    summary_status = (
        f"Target: <= 2.00%  ({n_evaluated} evaluated, {n_skipped} skipped)"
        if target_ok
        else f"EXCEEDS TARGET  ({n_evaluated} evaluated, {n_skipped} skipped)"
    )
    summary = pd.DataFrame([{
        "constraint_id": "OVERALL_ACTIVE_MEAN",
        "tier": np.nan,
        "rfc": "—",
        "description": "Mean FPR across constraints that were actually evaluated",
        "required_features": "",
        "missing_features": "",
        "n_rows": n,
        "n_violations": np.nan,
        "fpr": mean_fpr_active,
        "fpr_pct": mean_fpr_active * 100.0,
        "fpr_ci_lo_pct": np.nan,
        "fpr_ci_hi_pct": np.nan,
        "status": summary_status,
    }])
    return pd.concat([out, summary], ignore_index=True)


def print_calibration_table(cal: pd.DataFrame) -> None:
    """Human-readable view of the calibration table."""
    print("\n" + "=" * 100)
    print(f"{'ID':<28}{'Tier':<6}{'FPR %':>10}  {'95% CI %':<18} {'Status'}")
    print("=" * 100)
    for _, r in cal.iterrows():
        tier = "—" if pd.isna(r["tier"]) else str(int(r["tier"]))
        if pd.isna(r["fpr_pct"]):
            fpr_str = "       —"
        else:
            fpr_str = f"{r['fpr_pct']:>10.4f}"
        if pd.isna(r["fpr_ci_lo_pct"]):
            ci = ""
        else:
            ci = f"[{r['fpr_ci_lo_pct']:>6.3f}, {r['fpr_ci_hi_pct']:>6.3f}]"
        print(f"{r['constraint_id']:<28}{tier:<6}{fpr_str}  {ci:<18} {r['status']}")
    print("=" * 100 + "\n")
