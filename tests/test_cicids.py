"""
CIC-IDS-2017 pipeline smoke tests
==================================
Tests run without real CIC-IDS-2017 files.
Verifies column normalisation, constraint logic, and loader on
synthetic CICFlowMeter-shaped data.

Run:
    python tests/test_cicids.py
"""

from __future__ import annotations
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from grammar.data_loader_cicids import (
    _normalise_colname, _load_one_csv,
)
from grammar.constraints_cicids import (
    cc01_non_neg_duration, cc02_non_neg_bytes, cc03_non_neg_pkts,
    cc04_bytes_geq_pkts, cc05_fwd_mean_consistency,
    CICIDS_CONSTRAINTS,
)


def _check(label: str, cond: bool, detail: str = "") -> bool:
    flag = "PASS" if cond else "FAIL"
    extra = f"  -- {detail}" if detail else ""
    print(f"  {flag}  {label}{extra}")
    return cond


def _make_row(dur=1.0, fwd_bytes=1500, bwd_bytes=3000,
              fwd_pkts=10, bwd_pkts=20,
              fwd_mean=150.0, label="benign") -> dict:
    return {
        "Flow Duration":                     dur,
        "Total Length of Fwd Packets":       fwd_bytes,
        "Total Length of Bwd Packets":       bwd_bytes,
        "Total Fwd Packets":                 fwd_pkts,
        "Total Backward Packets":            bwd_pkts,
        "Fwd Packet Length Mean":            fwd_mean,
        "Bwd Packet Length Mean":            bwd_bytes / bwd_pkts if bwd_pkts else 0,
        "Flow Bytes/s":                      (fwd_bytes + bwd_bytes) / max(dur, 1e-9),
        "Flow IAT Mean":                     0.01,
        " Label":                            label,
    }


def _to_df(rows: list[dict]) -> pd.DataFrame:
    """Mimic what _load_one_csv does — normalise column names and add label_binary."""
    df = pd.DataFrame(rows)
    df.columns = [_normalise_colname(c) for c in df.columns]
    df["label_binary"] = (df["label"] != "benign").astype(int)
    return df


def test_column_normalisation() -> bool:
    print("\ntest_column_normalisation ...")
    ok = True
    cases = [
        (" Label",                       "label"),
        ("Flow Duration",                "flow_duration"),
        ("Flow Bytes/s",                 "flow_bytes_per_s"),
        ("Fwd Header Length.1",          "fwd_header_length_1"),
        ("Total Length of Fwd Packets",  "total_length_of_fwd_packets"),
        ("Init_Win_bytes_forward",       "init_win_bytes_forward"),
    ]
    for raw, expected in cases:
        got = _normalise_colname(raw)
        ok &= _check(f"'{raw}' → '{expected}'", got == expected, f"got '{got}'")
    return ok


def test_constraints() -> bool:
    print("\ntest_constraints ...")
    ok = True

    # CC01 non-neg duration
    df = _to_df([_make_row(dur=0.0), _make_row(dur=-0.001), _make_row(dur=5.0)])
    res = cc01_non_neg_duration(df)
    ok &= _check("CC01 valid/invalid/valid", res.tolist() == [True, False, True])

    # CC02 non-neg bytes
    df = _to_df([_make_row(fwd_bytes=0), _make_row(fwd_bytes=-10)])
    res = cc02_non_neg_bytes(df)
    ok &= _check("CC02 valid/invalid", res.tolist() == [True, False])

    # CC03 non-neg pkts
    df = _to_df([_make_row(fwd_pkts=1), _make_row(fwd_pkts=-1)])
    res = cc03_non_neg_pkts(df)
    ok &= _check("CC03 valid/invalid", res.tolist() == [True, False])

    # CC04 bytes >= pkts
    df = _to_df([
        _make_row(fwd_bytes=1500, fwd_pkts=10),
        _make_row(fwd_bytes=5, fwd_pkts=10),    # bytes < pkts — invalid
    ])
    res = cc04_bytes_geq_pkts(df)
    ok &= _check("CC04 valid/invalid", res.tolist() == [True, False])

    # CC05 fwd mean consistency
    df = _to_df([
        _make_row(fwd_mean=150.0, fwd_pkts=10, fwd_bytes=1500),   # exact
        _make_row(fwd_mean=300.0, fwd_pkts=10, fwd_bytes=1500),   # way off
        _make_row(fwd_mean=0.0, fwd_pkts=0, fwd_bytes=0),         # zero-pkt valid
    ])
    res = cc05_fwd_mean_consistency(df)
    ok &= _check("CC05 valid/invalid/valid (zero-pkt)", res.tolist() == [True, False, True])

    return ok


def test_loader_on_synthetic_csv() -> bool:
    """Build a fake _ISCX.csv, load it, verify column normalisation + label."""
    print("\ntest_loader_on_synthetic_csv ...")
    ok = True

    tmp = Path(tempfile.mkdtemp(prefix="pgcav_cicids_"))
    try:
        rows = [_make_row(label="benign")] * 30 + [_make_row(label="DoS Hulk")] * 10
        df_raw = pd.DataFrame(rows)
        fpath = tmp / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        df_raw.to_csv(fpath, index=False)

        df = _load_one_csv(fpath)
        ok &= _check("label column normalised", "label" in df.columns)
        ok &= _check("label_binary added", "label_binary" in df.columns)
        ok &= _check("benign rows = 30", int((df["label_binary"]==0).sum()) == 30,
                     f"got {(df['label_binary']==0).sum()}")
        ok &= _check("attack rows = 10", int((df["label_binary"]==1).sum()) == 10)
        ok &= _check("flow_duration column present",
                     "flow_duration" in df.columns)
        ok &= _check("flow_bytes_per_s normalised",
                     "flow_bytes_per_s" in df.columns)
        ok &= _check("no Inf in numerics", not df.select_dtypes(
            include=[float]).isin([np.inf, -np.inf]).any().any())

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return ok


def test_calibrate_on_synthetic() -> bool:
    """End-to-end: calibrate constraints on synthetic benign DataFrame."""
    print("\ntest_calibrate_on_synthetic ...")
    ok = True

    rows = [_make_row() for _ in range(500)]
    rows += [_make_row(fwd_bytes=-1) for _ in range(10)]  # 10 CC02 violations
    df = _to_df(rows)
    benign = df[df["label_binary"] == 0].copy()  # all synthetic = benign

    from experiments.e7_cicids_calibration import calibrate_cicids
    cal = calibrate_cicids(benign, bootstrap=False)

    ok &= _check("calibration returns DataFrame", isinstance(cal, pd.DataFrame))
    ok &= _check(f"correct number of rows", len(cal) == len(CICIDS_CONSTRAINTS),
                 f"got {len(cal)}")

    cc02_row = cal[cal["constraint_id"] == "CC02_non_neg_bytes"].iloc[0]
    expected = 10 / 510
    ok &= _check("CC02 FPR computed correctly",
                 abs(cc02_row["fpr"] - expected) < 1e-9,
                 f"expected {expected:.6f} got {cc02_row['fpr']:.6f}")

    return ok


def main() -> int:
    print("Running CIC-IDS-2017 pipeline smoke tests ...")
    print("-" * 60)
    ok = True
    ok &= test_column_normalisation()
    ok &= test_constraints()
    ok &= test_loader_on_synthetic_csv()
    ok &= test_calibrate_on_synthetic()
    print("-" * 60)
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
