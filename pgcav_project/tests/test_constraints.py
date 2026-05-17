"""
Unit tests for the 12 constraints.
==================================
These tests use small synthetic DataFrames that mimic the UNSW-NB15
schema. They verify each constraint correctly identifies valid vs
invalid rows BEFORE you run on the real dataset.

Run:
    python -m tests.test_constraints
or:
    pytest tests/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from grammar.constraints import (
    c01_ttl_range, c02_tcp_window, c03_tcp_seq, c04_non_neg_dur,
    c05_non_neg_timing, c06_non_neg_bytes, c07_non_neg_pkts,
    c08_bytes_geq_pkts, c09_loss_leq_pkts, c10_smean_consistency,
    c11_tcp_handshake_timing, c12_non_neg_jitter,
    CONSTRAINTS,
)
from grammar.validator import validate_dataframe, calibrate_constraints, validate_row


def _make_clean_row(**overrides) -> dict:
    """A protocol-valid TCP row that every constraint should accept."""
    row = {
        "proto": "tcp",
        "sttl": 64, "dttl": 128,
        "swin": 8192, "dwin": 8192,
        "stcpb": 100_000_000, "dtcpb": 200_000_000,
        "dur": 1.5,
        "tcprtt": 0.030, "synack": 0.015, "ackdat": 0.015,
        "sbytes": 1500, "dbytes": 3000,
        "spkts": 10, "dpkts": 20,
        "sloss": 0, "dloss": 0,
        "smeansz": 150,  # 150 * 10 = 1500 = sbytes
        "sjit": 0.001, "djit": 0.002,
    }
    row.update(overrides)
    return row


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _check(name, result_series, expected_list):
    got = result_series.tolist()
    ok = got == expected_list
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag}  {name:<32} expected={expected_list}  got={got}")
    return ok


def test_all() -> bool:
    print("Running constraint unit tests on synthetic data...")
    print("-" * 70)
    all_ok = True

    # ── C01 TTL ─────────────────────────────────────────────────────
    df = _df([
        _make_clean_row(),                       # valid
        _make_clean_row(sttl=0),                 # invalid: TTL=0
        _make_clean_row(dttl=256),               # invalid: > 255
        _make_clean_row(sttl=-5),                # invalid: negative
    ])
    all_ok &= _check("C01 ttl_range", c01_ttl_range(df), [True, False, False, False])

    # ── C02 TCP window (only enforced on tcp) ───────────────────────
    df = _df([
        _make_clean_row(proto="tcp", swin=8192),
        _make_clean_row(proto="tcp", swin=70000),    # > 65535
        _make_clean_row(proto="udp", swin=70000),    # ok because not TCP
    ])
    all_ok &= _check("C02 tcp_window", c02_tcp_window(df), [True, False, True])

    # ── C03 TCP seq ─────────────────────────────────────────────────
    df = _df([
        _make_clean_row(stcpb=0),
        _make_clean_row(stcpb=(1 << 32) - 1),        # boundary, valid
        _make_clean_row(stcpb=(1 << 32)),            # invalid: overflow
        _make_clean_row(dtcpb=-1),                   # invalid: negative
    ])
    all_ok &= _check("C03 tcp_seq", c03_tcp_seq(df), [True, True, False, False])

    # ── C04 non-neg duration ────────────────────────────────────────
    df = _df([
        _make_clean_row(dur=0.0),
        _make_clean_row(dur=10.0),
        _make_clean_row(dur=-0.001),
    ])
    all_ok &= _check("C04 non_neg_dur", c04_non_neg_dur(df), [True, True, False])

    # ── C05 non-neg timing ──────────────────────────────────────────
    df = _df([
        _make_clean_row(),
        _make_clean_row(tcprtt=-0.001),
        _make_clean_row(synack=-0.5),
    ])
    all_ok &= _check("C05 non_neg_timing", c05_non_neg_timing(df), [True, False, False])

    # ── C06 non-neg bytes ───────────────────────────────────────────
    df = _df([_make_clean_row(), _make_clean_row(sbytes=-1)])
    all_ok &= _check("C06 non_neg_bytes", c06_non_neg_bytes(df), [True, False])

    # ── C07 non-neg pkts ────────────────────────────────────────────
    df = _df([_make_clean_row(), _make_clean_row(spkts=-3)])
    all_ok &= _check("C07 non_neg_pkts", c07_non_neg_pkts(df), [True, False])

    # ── C08 bytes >= pkts ───────────────────────────────────────────
    df = _df([
        _make_clean_row(sbytes=1500, spkts=10),
        _make_clean_row(sbytes=5, spkts=10),     # fewer bytes than packets
    ])
    all_ok &= _check("C08 bytes_geq_pkts", c08_bytes_geq_pkts(df), [True, False])

    # ── C09 loss <= pkts ────────────────────────────────────────────
    df = _df([
        _make_clean_row(sloss=2, spkts=10),
        _make_clean_row(sloss=11, spkts=10),     # more loss than packets
    ])
    all_ok &= _check("C09 loss_leq_pkts", c09_loss_leq_pkts(df), [True, False])

    # ── C10 smean consistency ───────────────────────────────────────
    df = _df([
        _make_clean_row(sbytes=1500, spkts=10, smeansz=150),  # exact
        _make_clean_row(sbytes=1500, spkts=10, smeansz=151),  # within 5%
        _make_clean_row(sbytes=1500, spkts=10, smeansz=300),  # way off
        _make_clean_row(sbytes=0, spkts=0, smeansz=0),        # zero-pkt, valid
    ])
    all_ok &= _check("C10 smean_consistency", c10_smean_consistency(df),
                     [True, True, False, True])

    # ── C11 TCP handshake timing ────────────────────────────────────
    # TCPRTT_ABS_TOLERANCE = 0.01, so diff must exceed 10 ms to be invalid.
    df = _df([
        _make_clean_row(tcprtt=0.030, synack=0.015, ackdat=0.015),     # exact match
        _make_clean_row(tcprtt=0.100, synack=0.010, ackdat=0.010),     # off by 80 ms
        _make_clean_row(proto="udp", tcprtt=0.030, synack=0.0, ackdat=0.0),  # non-TCP, ignored
        _make_clean_row(tcprtt=0.0, synack=0.0, ackdat=0.0),           # all-zero, accepted
    ])
    all_ok &= _check("C11 tcp_handshake_timing", c11_tcp_handshake_timing(df),
                     [True, False, True, True])

    # ── C12 jitter non-neg ──────────────────────────────────────────
    df = _df([_make_clean_row(), _make_clean_row(sjit=-0.001)])
    all_ok &= _check("C12 non_neg_jitter", c12_non_neg_jitter(df), [True, False])

    # ── End-to-end: validate_dataframe ──────────────────────────────
    df = _df([_make_clean_row(), _make_clean_row(sttl=0, dur=-1)])
    out = validate_dataframe(df)
    all_ok &= _check("validate_dataframe all_valid",
                     out["all_valid"], [True, False])

    # ── End-to-end: validate_row ────────────────────────────────────
    r1 = validate_row(_make_clean_row())
    r2 = validate_row(_make_clean_row(sttl=0))
    ok = r1["valid"] is True and r2["valid"] is False
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag}  validate_row single row")
    all_ok &= ok

    # ── End-to-end: calibrate_constraints on a synthetic dataset ────
    # 1000 valid rows + 50 invalid rows → expected FPR for c01 = 50/1050
    rows = [_make_clean_row() for _ in range(1000)]
    rows += [_make_clean_row(sttl=0) for _ in range(50)]
    cal = calibrate_constraints(_df(rows), bootstrap=False)
    c01_row = cal[cal["constraint_id"] == "C01_ttl_range"].iloc[0]
    expected_fpr = 50.0 / 1050.0
    ok = abs(c01_row["fpr"] - expected_fpr) < 1e-9
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag}  calibrate_constraints C01 FPR  expected={expected_fpr:.6f}  got={c01_row['fpr']:.6f}")
    all_ok &= ok

    # ── Missing-column flagging: drop sttl/dttl and verify C01 is N/A ──
    rows = [_make_clean_row() for _ in range(100)]
    df_no_ttl = _df(rows).drop(columns=["sttl", "dttl"])
    cal_missing = calibrate_constraints(df_no_ttl, bootstrap=False)
    c01_skip = cal_missing[cal_missing["constraint_id"] == "C01_ttl_range"].iloc[0]
    skip_ok = (
        pd.isna(c01_skip["fpr"])
        and "N/A" in c01_skip["status"]
        and "sttl" in c01_skip["status"] and "dttl" in c01_skip["status"]
    )
    flag = "PASS" if skip_ok else "FAIL"
    print(f"  {flag}  missing sttl/dttl flagged as N/A  status='{c01_skip['status']}'")
    all_ok &= skip_ok

    # The summary row should report n_evaluated, n_skipped
    summary_row = cal_missing[cal_missing["constraint_id"] == "OVERALL_ACTIVE_MEAN"].iloc[0]
    sum_ok = "1 skipped" in summary_row["status"] or "skipped" in summary_row["status"]
    flag = "PASS" if sum_ok else "FAIL"
    print(f"  {flag}  summary row reports n_skipped  status='{summary_row['status']}'")
    all_ok &= sum_ok

    print("-" * 70)
    print(f"OVERALL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


if __name__ == "__main__":
    ok = test_all()
    sys.exit(0 if ok else 1)
