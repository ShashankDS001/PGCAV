"""
run_paper1.py
=============
Single-command end-to-end driver for the Paper 1 pipeline.

Steps performed:
    1. EDA / feature audit              -> results/tables/eda_feature_stats.csv
    2. Grammar calibration              -> results/tables/calibration_table.csv
    3. Figure generation (Figures 1-3)  -> results/figures/*.png

Usage:
    python run_paper1.py                 # default: run all three steps
    python run_paper1.py --skip-eda      # skip step 1 (e.g. after first run)
    python run_paper1.py --only figures  # only regenerate the figures

Prerequisites:
    * UNSW-NB15 CSVs in data/raw/ (see README.md for download URL)
    * `pip install -r requirements.txt`
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import e1_eda, e2_calibration, e3_figures


STEPS = {
    "eda": ("Feature audit (EDA)", e1_eda.main),
    "calibration": ("Grammar calibration", e2_calibration.main),
    "figures": ("Paper figures", e3_figures.main),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Paper 1 pipeline end-to-end.")
    p.add_argument("--skip-eda", action="store_true", help="skip the EDA step")
    p.add_argument("--skip-calibration", action="store_true",
                   help="skip calibration (figures will need an existing table)")
    p.add_argument("--skip-figures", action="store_true", help="skip figure generation")
    p.add_argument("--only", choices=list(STEPS.keys()),
                   help="run only one step")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.only:
        plan = [args.only]
    else:
        plan = []
        if not args.skip_eda:
            plan.append("eda")
        if not args.skip_calibration:
            plan.append("calibration")
        if not args.skip_figures:
            plan.append("figures")

    print("\n" + "#" * 70)
    print("# PGCAV Paper 1 — End-to-end pipeline")
    print("# Steps: " + ", ".join(plan))
    print("#" * 70 + "\n")

    overall_t0 = time.time()
    for step in plan:
        title, fn = STEPS[step]
        print(f"\n### Step: {title}\n")
        step_t0 = time.time()
        try:
            fn()
        except FileNotFoundError as e:
            print(e)
            return 2
        except Exception as e:
            print(f"\n[ERROR in step '{step}']  {type(e).__name__}: {e}")
            raise
        print(f"\n   step '{step}' done in {time.time()-step_t0:.1f}s")

    print(f"\nAll done in {time.time()-overall_t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
