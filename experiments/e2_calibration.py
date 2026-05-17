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

    # Partition by status
    skipped_mask    = individual["status"].str.startswith("N/A")
    informational   = individual[individual["status"] == "Demoted to Informational"]
    active          = individual[
        ~skipped_mask & ~individual["status"].eq("Demoted to Informational")
    ]
    n_total       = len(individual)
    n_evaluated   = len(individual[~skipped_mask])
    n_skipped     = len(individual[skipped_mask])
    n_informational = len(informational)
    n_active      = len(active)

    import numpy as np
    max_active_fpr = (
        float(np.nanmax(active["fpr"].values)) if n_active > 0 else float("nan")
    )

    mean_ok  = (n_active > 0 and overall_row["fpr"] <= GRAMMAR_FPR_ACTIVE_LIMIT)
    indiv_ok = (n_active > 0 and max_active_fpr <= GRAMMAR_FPR_TIER2_LIMIT)

    print(f"  Constraints evaluated  : {n_evaluated}/{n_total}  "
          f"(skipped: {n_skipped})")
    print(f"  Active constraints     : {n_active}   "
          f"Informational (excluded from gate): {n_informational}")
    print(f"  Mean Active FPR        : {overall_row['fpr_pct']:.4f}%  "
          f"(target <= {GRAMMAR_FPR_ACTIVE_LIMIT*100:.2f}%) -> "
          f"{'PASS' if mean_ok else 'FAIL'}")
    if n_active > 0:
        print(f"  Max Active FPR         : {max_active_fpr*100:.4f}%  "
              f"(target <= {GRAMMAR_FPR_TIER2_LIMIT*100:.2f}%) -> "
              f"{'PASS' if indiv_ok else 'FAIL'}")
    else:
        print(f"  Max Active FPR         : N/A — no Active constraints")

    if n_informational > 0:
        print(f"\n  Informational (gate-excluded) constraints:")
        for _, row in informational.iterrows():
            print(f"    {row['constraint_id']:<30} FPR={row['fpr_pct']:.4f}%  "
                  f"({row['status']})")

    # ── Defensibility warnings ───────────────────────────────────────
    warnings = []

    tier1_skipped = individual[skipped_mask & (individual["tier"] == 1)]
    if len(tier1_skipped) > 0:
        warnings.append(
            f"  TIER-1 CONSTRAINTS SKIPPED: "
            f"{', '.join(tier1_skipped['constraint_id'].tolist())}\n"
            f"    Fix: download Format A (4-CSV raw) which retains all 49 features."
        )

    if n_skipped > n_total / 2:
        warnings.append(
            f"  MAJORITY SKIPPED ({n_skipped}/{n_total}): "
            f"mean FPR is computed over a minority of constraints."
        )

    if n_active >= 5:
        n_zero = int((active["fpr"] == 0).sum())
        if n_zero / n_active >= 0.75:
            warnings.append(
                f"  {n_zero}/{n_active} Active constraints have FPR = 0.0000%.\n"
                f"    This is expected when traffic stays well within RFC bounds.\n"
                f"    Discriminating power is validated against adversarial examples."
            )

    if warnings:
        print("\n" + "=" * 70)
        print("NOTES — review before finalising the manuscript")
        print("=" * 70)
        for w in warnings:
            print("\n" + w)

    print()
    if n_informational > 0 and mean_ok and indiv_ok:
        print(f"  >>> Paper 1 calibration PASSES.")
        print(f"      {n_informational} constraint(s) demoted to Informational")
        print(f"      (excluded from gate — see note in manuscript).")
    elif mean_ok and indiv_ok:
        print("  >>> Paper 1 calibration PASSES cleanly. Proceed to figures.")
    else:
        print("  >>> Calibration FAILS the gate.")
        print("      Inspect the table and adjust bounds or tolerances in config.py.")

    return out_path


if __name__ == "__main__":
    main()