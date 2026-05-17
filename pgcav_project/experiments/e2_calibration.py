"""
Experiment 2 — Grammar Calibration (Paper 1 main result)
=========================================================
Run all 12 constraints against ~900k clean UNSW-NB15 normal flows.
Produce the calibration table (Table 2 in the paper) with per-constraint
FPR plus 95% bootstrap confidence intervals.

Go/No-Go gate (Doc2 §3.3):
    * Mean FPR across Active constraints <= 2.00%
    * No individual constraint exceeds 5.00%

Run:
    python -m experiments.e2_calibration
or via run_paper1.py.
"""

from __future__ import annotations
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RESULTS_TABLES,
    GRAMMAR_FPR_ACTIVE_LIMIT,
    GRAMMAR_FPR_TIER2_LIMIT,
)
from grammar.data_loader import load_unsw_normal
from grammar.validator import calibrate_constraints, print_calibration_table


def main() -> Path:
    print("=" * 70)
    print("E2 — Grammar calibration on clean UNSW-NB15 traffic")
    print("=" * 70)

    t0 = time.time()
    normal = load_unsw_normal()
    print(f"Loaded {len(normal):,} clean rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    cal = calibrate_constraints(normal, bootstrap=True)
    print(f"Calibration finished ({time.time()-t0:.1f}s)")

    out_path = RESULTS_TABLES / "calibration_table.csv"
    cal.to_csv(out_path, index=False)
    print(f"\nSaved calibration table to: {out_path}")

    print_calibration_table(cal)

    # ── Go / No-Go gate ─────────────────────────────────────────────
    print("=" * 70)
    print("GO / NO-GO GATE (Doc2 §3.3)")
    print("=" * 70)

    overall_row = cal[cal["constraint_id"] == "OVERALL_ACTIVE_MEAN"].iloc[0]
    individual = cal[cal["constraint_id"] != "OVERALL_ACTIVE_MEAN"]

    # Distinguish evaluated from skipped
    skipped_mask = individual["status"].str.startswith("N/A")
    evaluated = individual[~skipped_mask]
    skipped = individual[skipped_mask]
    n_total = len(individual)
    n_evaluated = len(evaluated)
    n_skipped = len(skipped)

    # Use nanmax to ignore NaN rows
    import numpy as np
    if n_evaluated > 0:
        max_individual_fpr = float(np.nanmax(evaluated["fpr"].values))
    else:
        max_individual_fpr = float("nan")

    mean_ok = (n_evaluated > 0 and overall_row["fpr"] <= GRAMMAR_FPR_ACTIVE_LIMIT)
    indiv_ok = (n_evaluated > 0 and max_individual_fpr <= GRAMMAR_FPR_TIER2_LIMIT)

    print(f"  Constraints evaluated  : {n_evaluated}/{n_total}"
          f"   (skipped: {n_skipped})")
    print(f"  Mean Active FPR        : {overall_row['fpr_pct']:.4f}%  "
          f"(target <= {GRAMMAR_FPR_ACTIVE_LIMIT*100:.2f}%) -> "
          f"{'PASS' if mean_ok else 'FAIL'}")
    if n_evaluated > 0:
        print(f"  Max individual FPR     : {max_individual_fpr*100:.4f}%  "
              f"(target <= {GRAMMAR_FPR_TIER2_LIMIT*100:.2f}%) -> "
              f"{'PASS' if indiv_ok else 'FAIL'}")
    else:
        print(f"  Max individual FPR     : N/A — no constraints evaluated")

    # ── Critical defensibility warnings ─────────────────────────────
    warnings = []

    # Warning A: any Tier-1 (RFC-direct) skipped is a paper-killer
    tier1_skipped = skipped[skipped["tier"] == 1]
    if len(tier1_skipped) > 0:
        warnings.append(
            f"  TIER-1 CONSTRAINTS SKIPPED:  "
            f"{', '.join(tier1_skipped['constraint_id'].tolist())}\n"
            f"    These are RFC-direct and central to the grammar's defensibility.\n"
            f"    Cause: required columns missing from your UNSW-NB15 distribution.\n"
            f"    Fix:   download the Format A (4-CSV raw) version which retains\n"
            f"           all 49 features including sttl, dttl."
        )

    # Warning B: more than half skipped
    if n_skipped > n_total / 2:
        warnings.append(
            f"  MAJORITY OF CONSTRAINTS SKIPPED ({n_skipped}/{n_total}):\n"
            f"    The reported mean FPR is computed over only the evaluated\n"
            f"    constraints. A reviewer will read this as a near-vacuous test.\n"
            f"    Cause: your dataset variant has dropped or transformed features.\n"
            f"    Fix:   download the 4-CSV raw version into data/raw/."
        )

    # Warning C: anomalously low FPR across many constraints
    # (suggests vacuous bounds, not real validation power)
    if n_evaluated >= 5:
        n_zero_fpr = int((evaluated["fpr"] == 0).sum())
        if n_zero_fpr / n_evaluated >= 0.75:
            warnings.append(
                f"  {n_zero_fpr}/{n_evaluated} evaluated constraints have FPR = 0.0000%:\n"
                f"    Could indicate vacuous bounds (the data range never approaches\n"
                f"    the constraint threshold). Inspect the EDA feature stats to\n"
                f"    verify the constraints are doing real work."
            )

    if warnings:
        print("\n" + "=" * 70)
        print("DEFENSIBILITY WARNINGS — read these before claiming PASS")
        print("=" * 70)
        for w in warnings:
            print("\n" + w)

    print()
    if mean_ok and indiv_ok and not tier1_skipped.empty:
        print("  >>> Gates technically PASS but Tier-1 constraints were skipped.")
        print("      Do not write the paper on this calibration alone.")
    elif mean_ok and indiv_ok and warnings:
        print("  >>> Gates technically PASS but defensibility warnings raised.")
        print("      Read the warnings above before proceeding to manuscript.")
    elif mean_ok and indiv_ok:
        print("  >>> Paper 1 calibration PASSES cleanly. Proceed to figures.")
    else:
        print("  >>> Calibration FAILS the gate.")
        print("      Inspect the table above and adjust constraint bounds")
        print("      or tolerances in config.py, then re-run.")

    return out_path


if __name__ == "__main__":
    main()
