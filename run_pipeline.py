"""
run_pipeline.py
===============
Unified driver for everything built so far:

  Phase 1 (Paper 1):  grammar construction + calibration on clean traffic
      step 'eda'           -> results/tables/eda_feature_stats.csv
      step 'calibration'   -> results/tables/calibration_table.csv
      step 'figures'       -> results/figures/fig{1,2,3}_*.png

  Phase 2A (Paper 2 prep): classifier training and baseline evaluation
      step 'classifiers'   -> models/{preprocessor,rf,mlp}.pkl
                              results/tables/classifier_baseline.csv

Usage:
    python run_pipeline.py                          # run everything in order
    python run_pipeline.py --only calibration       # run a single step
    python run_pipeline.py --skip classifiers       # skip one step

The original run_paper1.py still works for Phase 1 only.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import (
    e1_eda, e2_calibration, e3_figures, e4_train_classifiers,
    e5_generate_adversarials, e6_detection_evaluation,
)


STEPS = [
    ("eda",          "Feature audit (EDA)",                e1_eda.main),
    ("calibration",  "Grammar calibration",                e2_calibration.main),
    ("figures",      "Paper 1 figures",                    e3_figures.main),
    ("classifiers",  "Train RF + MLP + baseline metrics",  e4_train_classifiers.main),
    ("adversarials", "Generate adversarial corpora",       e5_generate_adversarials.main),
    ("detection",    "Grammar-as-Detector evaluation",     e6_detection_evaluation.main),
]
STEP_NAMES = [s[0] for s in STEPS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the PGCAV pipeline end-to-end.")
    p.add_argument("--only", choices=STEP_NAMES, action="append",
                   help="run only this step (can be given multiple times)")
    p.add_argument("--skip", choices=STEP_NAMES, action="append",
                   help="skip this step (can be given multiple times)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.only:
        plan = [s for s in STEPS if s[0] in args.only]
    else:
        skip = set(args.skip or [])
        plan = [s for s in STEPS if s[0] not in skip]

    if not plan:
        print("Nothing to do. Pass --only or remove --skip flags.")
        return 0

    print("\n" + "#" * 70)
    print(f"# PGCAV pipeline — running: {', '.join(s[0] for s in plan)}")
    print("#" * 70)

    overall_t0 = time.time()
    for name, title, fn in plan:
        print(f"\n### Step '{name}': {title}\n")
        step_t0 = time.time()
        try:
            fn()
        except FileNotFoundError as e:
            print(e)
            return 2
        except Exception as e:
            print(f"\n[ERROR in step '{name}']  {type(e).__name__}: {e}")
            raise
        print(f"\n   step '{name}' done in {time.time()-step_t0:.1f}s")

    print(f"\nAll done in {time.time()-overall_t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
