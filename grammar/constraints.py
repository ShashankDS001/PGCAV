"""
PGCAV Protocol Constraint Grammar — 12 Constraints
===================================================
Each constraint is a vectorised function over a pandas DataFrame.
It returns a Boolean Series where:
    True  = row satisfies the constraint (VALID)
    False = row violates the constraint  (this is what we count as FPR
            when measured on clean normal traffic).

Tier 1 = RFC-direct constraints (numerical bounds from IETF specs).
Tier 2 = logical invariants derived from UNSW-NB15 feature definitions.

REFERENCES
----------
RFC 791  IP                       https://datatracker.ietf.org/doc/html/rfc791
RFC 793  TCP                      https://datatracker.ietf.org/doc/html/rfc793
RFC 768  UDP                      https://datatracker.ietf.org/doc/html/rfc768
UNSW-NB15 feature descriptions    Moustafa & Slay, MilCIS 2015

DESIGN NOTES
------------
* All constraints operate on lowercase column names (see data_loader.py).
* Missing columns return "all valid" — a conservative default that lets
  the calibration continue and flags the absence in the FPR row.
* Tolerance on Tier 2 floating-point constraints is configurable in
  config.py (SMEAN_REL_TOLERANCE, TCPRTT_ABS_TOLERANCE).
"""

from __future__ import annotations
from typing import Callable

import numpy as np
import pandas as pd

from config import SMEAN_REL_TOLERANCE, TCPRTT_ABS_TOLERANCE


# ── Helper: safely fetch a numeric column ─────────────────────────────
def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    """Return df[name] as float, or None if the column is missing."""
    if name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce")


def _all_valid(n: int) -> pd.Series:
    """Constant 'all rows valid' result when required columns are absent."""
    return pd.Series([True] * n, dtype=bool)


# ──────────────────────────────────────────────────────────────────────
# TIER 1 — RFC-direct constraints
# ──────────────────────────────────────────────────────────────────────

def c01_ttl_range(df: pd.DataFrame) -> pd.Series:
    """RFC 791 §3.1: IP Time-to-Live is an 8-bit unsigned field, [1, 255].

    A TTL of 0 means the packet should be discarded — so a flow recording
    a non-zero source TTL of 0 represents an unobservable physical state.
    """
    sttl = _col(df, "sttl")
    dttl = _col(df, "dttl")
    if sttl is None or dttl is None:
        return _all_valid(len(df))
    return sttl.between(1, 255) & dttl.between(1, 255)


def c02_tcp_window(df: pd.DataFrame) -> pd.Series:
    """RFC 793 §3.1: TCP window field is 16-bit unsigned, [0, 65535].

    For non-TCP flows the field is meaningless; we return True.
    """
    proto = df["proto"].astype(str).str.lower() if "proto" in df.columns else None
    swin = _col(df, "swin")
    dwin = _col(df, "dwin")
    if proto is None or swin is None or dwin is None:
        return _all_valid(len(df))
    is_tcp = proto.eq("tcp")
    valid = swin.between(0, 65535) & dwin.between(0, 65535)
    # Non-TCP rows are valid by default
    return ~is_tcp | valid


def c03_tcp_seq(df: pd.DataFrame) -> pd.Series:
    """RFC 793 §3.1: TCP sequence numbers are 32-bit unsigned, [0, 2^32-1]."""
    stcpb = _col(df, "stcpb")
    dtcpb = _col(df, "dtcpb")
    if stcpb is None or dtcpb is None:
        return _all_valid(len(df))
    max_seq = (1 << 32) - 1  # 4_294_967_295
    return stcpb.between(0, max_seq) & dtcpb.between(0, max_seq)


def c04_non_neg_dur(df: pd.DataFrame) -> pd.Series:
    """Derived: flow duration cannot be negative."""
    dur = _col(df, "dur")
    if dur is None:
        return _all_valid(len(df))
    return dur >= 0


def c12_non_neg_jitter(df: pd.DataFrame) -> pd.Series:
    """Derived: jitter is a magnitude; cannot be negative."""
    sjit = _col(df, "sjit")
    djit = _col(df, "djit")
    if sjit is None or djit is None:
        return _all_valid(len(df))
    return (sjit >= 0) & (djit >= 0)


# ──────────────────────────────────────────────────────────────────────
# TIER 2 — Logical invariants on UNSW-NB15 feature semantics
# ──────────────────────────────────────────────────────────────────────

def c05_non_neg_timing(df: pd.DataFrame) -> pd.Series:
    """tcprtt, synack, ackdat are timing values: cannot be negative."""
    tcprtt = _col(df, "tcprtt")
    synack = _col(df, "synack")
    ackdat = _col(df, "ackdat")
    if tcprtt is None or synack is None or ackdat is None:
        return _all_valid(len(df))
    return (tcprtt >= 0) & (synack >= 0) & (ackdat >= 0)


def c06_non_neg_bytes(df: pd.DataFrame) -> pd.Series:
    """Byte counts (sbytes, dbytes) cannot be negative."""
    sbytes = _col(df, "sbytes")
    dbytes = _col(df, "dbytes")
    if sbytes is None or dbytes is None:
        return _all_valid(len(df))
    return (sbytes >= 0) & (dbytes >= 0)


def c07_non_neg_pkts(df: pd.DataFrame) -> pd.Series:
    """Packet counts (spkts, dpkts) cannot be negative."""
    spkts = _col(df, "spkts")
    dpkts = _col(df, "dpkts")
    if spkts is None or dpkts is None:
        return _all_valid(len(df))
    return (spkts >= 0) & (dpkts >= 0)


def c08_bytes_geq_pkts(df: pd.DataFrame) -> pd.Series:
    """sbytes >= spkts and dbytes >= dpkts.

    Weak lower bound: each packet carries at least 1 byte. Tighter bounds
    (e.g. >= 20 bytes/packet for IP header) trigger high FPR on flows with
    zero-payload control packets, so we keep the conservative form.
    """
    sbytes = _col(df, "sbytes")
    dbytes = _col(df, "dbytes")
    spkts = _col(df, "spkts")
    dpkts = _col(df, "dpkts")
    if any(c is None for c in (sbytes, dbytes, spkts, dpkts)):
        return _all_valid(len(df))
    return (sbytes >= spkts) & (dbytes >= dpkts)


def c09_loss_leq_pkts(df: pd.DataFrame) -> pd.Series:
    """sloss <= spkts and dloss <= dpkts: lost packets ≤ total packets."""
    sloss = _col(df, "sloss")
    dloss = _col(df, "dloss")
    spkts = _col(df, "spkts")
    dpkts = _col(df, "dpkts")
    if any(c is None for c in (sloss, dloss, spkts, dpkts)):
        return _all_valid(len(df))
    return (sloss <= spkts) & (dloss <= dpkts)


def c10_smean_consistency(df: pd.DataFrame) -> pd.Series:
    """smeansz * spkts ≈ sbytes (within relative tolerance).

    UNSW-NB15 stores smeansz as a rounded integer, so we accept
    |smeansz*spkts - sbytes| <= max(1.0, REL_TOL * sbytes).

    Zero-packet flows are trivially valid (no packets ⇒ no mean).
    """
    smean = _col(df, "smeansz")
    spkts = _col(df, "spkts")
    sbytes = _col(df, "sbytes")
    if any(c is None for c in (smean, spkts, sbytes)):
        return _all_valid(len(df))
    predicted = smean * spkts
    tol = np.maximum(1.0, SMEAN_REL_TOLERANCE * sbytes.abs())
    valid = (predicted - sbytes).abs() <= tol
    # rows with zero packets cannot violate this constraint
    zero_pkts = spkts.eq(0)
    return valid | zero_pkts


def c11_tcp_handshake_timing(df: pd.DataFrame) -> pd.Series:
    """For TCP, tcprtt ≈ synack + ackdat (within absolute tolerance).

    For non-TCP flows the decomposition is not defined; we return True.
    """
    proto = df["proto"].astype(str).str.lower() if "proto" in df.columns else None
    tcprtt = _col(df, "tcprtt")
    synack = _col(df, "synack")
    ackdat = _col(df, "ackdat")
    if proto is None or tcprtt is None or synack is None or ackdat is None:
        return _all_valid(len(df))
    is_tcp = proto.eq("tcp")
    diff = (tcprtt - (synack + ackdat)).abs()
    valid = diff <= TCPRTT_ABS_TOLERANCE
    # also accept the (frequent) case where all three are zero on non-handshaked TCP
    all_zero = tcprtt.eq(0) & synack.eq(0) & ackdat.eq(0)
    return ~is_tcp | valid | all_zero


# ──────────────────────────────────────────────────────────────────────
# Constraint registry
# ──────────────────────────────────────────────────────────────────────

CONSTRAINTS: dict[str, dict] = {
    "C01_ttl_range": {
        "tier": 1,
        "fn": c01_ttl_range,
        "features": ["sttl", "dttl"],
        "description": "IP TTL in [1, 255]",
        "rfc": "RFC 791 §3.1",
    },
    "C02_tcp_window": {
        "tier": 1,
        "fn": c02_tcp_window,
        "features": ["swin", "dwin", "proto"],
        "description": "TCP window in [0, 65535] when proto=tcp",
        "rfc": "RFC 793 §3.1",
    },
    "C03_tcp_seq": {
        "tier": 1,
        "fn": c03_tcp_seq,
        "features": ["stcpb", "dtcpb"],
        "description": "TCP base-seq in [0, 2^32-1]",
        "rfc": "RFC 793 §3.1",
    },
    "C04_non_neg_dur": {
        "tier": 1,
        "fn": c04_non_neg_dur,
        "features": ["dur"],
        "description": "Flow duration >= 0",
        "rfc": "Derived",
    },
    "C12_non_neg_jitter": {
        "tier": 1,
        "fn": c12_non_neg_jitter,
        "features": ["sjit", "djit"],
        "description": "Jitter magnitudes >= 0",
        "rfc": "Derived",
    },
    "C05_non_neg_timing": {
        "tier": 2,
        "fn": c05_non_neg_timing,
        "features": ["tcprtt", "synack", "ackdat"],
        "description": "Timing values (tcprtt, synack, ackdat) >= 0",
        "rfc": "Derived",
    },
    "C06_non_neg_bytes": {
        "tier": 2,
        "fn": c06_non_neg_bytes,
        "features": ["sbytes", "dbytes"],
        "description": "Byte counts >= 0",
        "rfc": "Derived",
    },
    "C07_non_neg_pkts": {
        "tier": 2,
        "fn": c07_non_neg_pkts,
        "features": ["spkts", "dpkts"],
        "description": "Packet counts >= 0",
        "rfc": "Derived",
    },
    "C08_bytes_geq_pkts": {
        "tier": 2,
        "fn": c08_bytes_geq_pkts,
        "features": ["sbytes", "spkts", "dbytes", "dpkts"],
        "description": "sbytes >= spkts and dbytes >= dpkts",
        "rfc": "Derived",
    },
    "C09_loss_leq_pkts": {
        "tier": 2,
        "fn": c09_loss_leq_pkts,
        "features": ["sloss", "spkts", "dloss", "dpkts"],
        "description": "Loss counts <= packet counts",
        "rfc": "Derived",
    },
    "C10_smean_consistency": {
        "tier": 2,
        "fn": c10_smean_consistency,
        "features": ["smeansz", "spkts", "sbytes"],
        "description": "smeansz * spkts ≈ sbytes (within tolerance)",
        "rfc": "Derived",
    },
    "C11_tcp_handshake_timing": {
        "tier": 2,
        "fn": c11_tcp_handshake_timing,
        "features": ["tcprtt", "synack", "ackdat", "proto"],
        "description": "TCP: tcprtt ≈ synack + ackdat (within tolerance)",
        "rfc": "Derived",
    },
}

TIER1_CONSTRAINTS = {k: v for k, v in CONSTRAINTS.items() if v["tier"] == 1}
TIER2_CONSTRAINTS = {k: v for k, v in CONSTRAINTS.items() if v["tier"] == 2}


def get_constraint(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Look up a constraint function by ID."""
    if name not in CONSTRAINTS:
        raise KeyError(f"unknown constraint: {name}")
    return CONSTRAINTS[name]["fn"]
