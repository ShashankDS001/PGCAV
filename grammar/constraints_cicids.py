"""
PGCAV Grammar Constraints — CIC-IDS-2017 Edition
=================================================
CIC-IDS-2017 uses CICFlowMeter features — different names, several
UNSW-NB15 protocol-level fields absent. 5 of 12 constraints transfer.

Transferable (✅) vs not (❌):
    ✅ CC01  flow_duration >= 0             (≈ C04)
    ✅ CC02  total byte counts >= 0         (≈ C06)
    ✅ CC03  packet counts >= 0             (≈ C07)
    ✅ CC04  total bytes >= packet count    (≈ C08)
    ✅ CC05  fwd mean × pkts ≈ total bytes  (≈ C10)
    ❌ C01/C02/C03/C05/C09/C11/C12  columns absent from CICFlowMeter

Paper narrative:
    "UNSW-NB15 (raw NetFlow) supports 11/12 active constraints.
     CIC-IDS-2017 (CICFlowMeter) supports 5/12, confirming the grammar
     methodology generalises across feature extraction pipelines while
     coverage depends on feature depth."
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from config import SMEAN_REL_TOLERANCE


def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce")


def _all_valid(n: int) -> pd.Series:
    return pd.Series([True] * n, dtype=bool)


def cc01_non_neg_duration(df: pd.DataFrame) -> pd.Series:
    """Flow duration >= 0."""
    c = _col(df, "flow_duration")
    return _all_valid(len(df)) if c is None else (c >= 0)


def cc02_non_neg_bytes(df: pd.DataFrame) -> pd.Series:
    """Total payload bytes >= 0 in both directions.

    CICFlowMeter integer-overflow bug produces negative values.
    This constraint correctly flags those rows.
    """
    fwd = _col(df, "total_length_of_fwd_packets")
    bwd = _col(df, "total_length_of_bwd_packets")
    if fwd is None or bwd is None:
        return _all_valid(len(df))
    return (fwd >= 0) & (bwd >= 0)


def cc03_non_neg_pkts(df: pd.DataFrame) -> pd.Series:
    """Packet counts >= 0 in both directions."""
    fwd = _col(df, "total_fwd_packets")
    bwd = _col(df, "total_backward_packets")
    if fwd is None or bwd is None:
        return _all_valid(len(df))
    return (fwd >= 0) & (bwd >= 0)


def cc04_bytes_geq_pkts(df: pd.DataFrame) -> pd.Series:
    """Total bytes >= packet count (minimum 1 byte per packet)."""
    fb = _col(df, "total_length_of_fwd_packets")
    fp = _col(df, "total_fwd_packets")
    bb = _col(df, "total_length_of_bwd_packets")
    bp = _col(df, "total_backward_packets")
    if any(c is None for c in (fb, fp, bb, bp)):
        return _all_valid(len(df))
    return (fb >= fp) & (bb >= bp)


def cc05_fwd_mean_consistency(df: pd.DataFrame) -> pd.Series:
    """fwd_packet_length_mean × total_fwd_packets ≈ total_length_of_fwd_packets."""
    mean  = _col(df, "fwd_packet_length_mean")
    pkts  = _col(df, "total_fwd_packets")
    total = _col(df, "total_length_of_fwd_packets")
    if any(c is None for c in (mean, pkts, total)):
        return _all_valid(len(df))
    predicted = mean * pkts
    tol = np.maximum(1.0, SMEAN_REL_TOLERANCE * total.abs())
    return ((predicted - total).abs() <= tol) | pkts.eq(0)


CICIDS_CONSTRAINTS: dict[str, dict] = {
    "CC01_non_neg_duration": {
        "tier": 1, "fn": cc01_non_neg_duration,
        "features": ["flow_duration"],
        "description": "Flow duration >= 0",
        "unsw_equivalent": "C04",
    },
    "CC02_non_neg_bytes": {
        "tier": 2, "fn": cc02_non_neg_bytes,
        "features": ["total_length_of_fwd_packets", "total_length_of_bwd_packets"],
        "description": "Total bytes >= 0 (flags CICFlowMeter overflow bug)",
        "unsw_equivalent": "C06",
    },
    "CC03_non_neg_pkts": {
        "tier": 2, "fn": cc03_non_neg_pkts,
        "features": ["total_fwd_packets", "total_backward_packets"],
        "description": "Packet counts >= 0",
        "unsw_equivalent": "C07",
    },
    "CC04_bytes_geq_pkts": {
        "tier": 2, "fn": cc04_bytes_geq_pkts,
        "features": ["total_length_of_fwd_packets", "total_fwd_packets",
                     "total_length_of_bwd_packets", "total_backward_packets"],
        "description": "Bytes >= packet count per direction",
        "unsw_equivalent": "C08",
    },
    "CC05_fwd_mean_consistency": {
        "tier": 2, "fn": cc05_fwd_mean_consistency,
        "features": ["fwd_packet_length_mean", "total_fwd_packets",
                     "total_length_of_fwd_packets"],
        "description": "fwd mean pkt size × fwd pkt count ≈ total fwd bytes",
        "unsw_equivalent": "C10",
    },
}
