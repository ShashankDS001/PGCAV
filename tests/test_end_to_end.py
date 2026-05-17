"""
End-to-end integration test
============================
Generates a synthetic UNSW-NB15-shaped CSV with ~10k normal rows and
~3k attack rows, then runs the full Paper 1 pipeline (EDA + calibration
+ figures) against it. Verifies all output files are produced.

This is a sanity check before you run against the real ~2.5 M-row dataset.

Run:
    python tests/test_end_to_end.py
"""

from __future__ import annotations
import sys
import tempfile
import shutil
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from grammar.data_loader import _UNSW_COLUMNS_FALLBACK


def build_synthetic_unsw(n_normal: int = 10_000, n_attack: int = 3_000,
                        seed: int = 42) -> pd.DataFrame:
    """Generate a DataFrame with plausible UNSW-NB15 column ranges."""
    rng = np.random.default_rng(seed)
    n = n_normal + n_attack

    proto_choices = ["tcp", "udp", "icmp", "tcp", "tcp", "udp"]  # tcp-heavy
    service_choices = ["http", "dns", "smtp", "ftp", "-", "ssh"]
    state_choices = ["FIN", "INT", "CON", "REQ", "RST"]

    spkts = rng.integers(1, 200, size=n)
    dpkts = rng.integers(1, 200, size=n)
    smeansz = rng.integers(40, 1500, size=n)
    sbytes = (spkts * smeansz) + rng.integers(-50, 50, size=n).clip(min=0)
    dbytes = (dpkts * rng.integers(40, 1500, size=n)) + rng.integers(0, 100, size=n)

    df = pd.DataFrame({
        "srcip": ["10.0.0.1"] * n,
        "sport": rng.integers(1024, 65535, size=n),
        "dstip": ["10.0.0.2"] * n,
        "dsport": rng.integers(1, 65535, size=n),
        "proto": rng.choice(proto_choices, size=n),
        "state": rng.choice(state_choices, size=n),
        "dur": rng.uniform(0, 100, size=n),
        "sbytes": sbytes.clip(min=0),
        "dbytes": dbytes.clip(min=0),
        "sttl": rng.integers(32, 255, size=n),
        "dttl": rng.integers(32, 255, size=n),
        "sloss": rng.integers(0, 5, size=n).clip(max=spkts),
        "dloss": rng.integers(0, 5, size=n).clip(max=dpkts),
        "service": rng.choice(service_choices, size=n),
        "sload": rng.uniform(0, 1e6, size=n),
        "dload": rng.uniform(0, 1e6, size=n),
        "spkts": spkts,
        "dpkts": dpkts,
        "swin": rng.integers(0, 65535, size=n),
        "dwin": rng.integers(0, 65535, size=n),
        "stcpb": rng.integers(0, 4_000_000_000, size=n),
        "dtcpb": rng.integers(0, 4_000_000_000, size=n),
        "smeansz": smeansz,
        "dmeansz": rng.integers(40, 1500, size=n),
        "trans_depth": rng.integers(0, 5, size=n),
        "res_bdy_len": rng.integers(0, 1000, size=n),
        "sjit": rng.uniform(0, 0.05, size=n),
        "djit": rng.uniform(0, 0.05, size=n),
        "stime": rng.integers(1421927000, 1421930000, size=n),
        "ltime": rng.integers(1421930000, 1421940000, size=n),
        "sintpkt": rng.uniform(0, 0.1, size=n),
        "dintpkt": rng.uniform(0, 0.1, size=n),
        "tcprtt": rng.uniform(0, 0.1, size=n),
        "synack": rng.uniform(0, 0.05, size=n),
        "ackdat": rng.uniform(0, 0.05, size=n),
        "is_sm_ips_ports": rng.integers(0, 2, size=n),
        "ct_state_ttl": rng.integers(0, 10, size=n),
        "ct_flw_http_mthd": rng.integers(0, 5, size=n),
        "is_ftp_login": rng.integers(0, 2, size=n),
        "ct_ftp_cmd": rng.integers(0, 5, size=n),
        "ct_srv_src": rng.integers(0, 50, size=n),
        "ct_srv_dst": rng.integers(0, 50, size=n),
        "ct_dst_ltm": rng.integers(0, 20, size=n),
        "ct_src_ltm": rng.integers(0, 20, size=n),
        "ct_src_dport_ltm": rng.integers(0, 20, size=n),
        "ct_dst_sport_ltm": rng.integers(0, 20, size=n),
        "ct_dst_src_ltm": rng.integers(0, 20, size=n),
        "attack_cat": [""] * n_normal + ["DoS"] * n_attack,
        "label": [0] * n_normal + [1] * n_attack,
    })

    # Make tcprtt ≈ synack + ackdat for the C11 constraint
    df["tcprtt"] = df["synack"] + df["ackdat"]

    # Reorder to canonical UNSW order
    df = df[_UNSW_COLUMNS_FALLBACK]
    return df


def main() -> int:
    print("=" * 70)
    print("End-to-end integration test on synthetic UNSW-NB15-shaped data")
    print("=" * 70)

    tmp = Path(tempfile.mkdtemp(prefix="pgcav_e2e_"))
    try:
        # Generate synthetic CSV
        df = build_synthetic_unsw(n_normal=10_000, n_attack=3_000)
        csv_path = tmp / "UNSW-NB15_1.csv"
        df.to_csv(csv_path, header=False, index=False)
        print(f"  wrote synthetic CSV: {csv_path}  shape={df.shape}")

        # Patch config to point at temp dir
        original = {
            "DATA_RAW": config.DATA_RAW,
            "DATA_PROCESSED": config.DATA_PROCESSED,
            "RESULTS_TABLES": config.RESULTS_TABLES,
            "RESULTS_FIGS": config.RESULTS_FIGS,
            "PROCESSED_NORMAL_PARQUET": config.PROCESSED_NORMAL_PARQUET,
            "PROCESSED_FULL_PARQUET": config.PROCESSED_FULL_PARQUET,
            "UNSW_FILES": list(config.UNSW_FILES),
        }
        config.DATA_RAW = tmp
        config.DATA_PROCESSED = tmp / "processed"
        config.RESULTS_TABLES = tmp / "results_tables"
        config.RESULTS_FIGS = tmp / "results_figs"
        config.DATA_PROCESSED.mkdir(exist_ok=True)
        config.RESULTS_TABLES.mkdir(exist_ok=True)
        config.RESULTS_FIGS.mkdir(exist_ok=True)
        config.PROCESSED_NORMAL_PARQUET = config.DATA_PROCESSED / "n.parquet"
        config.PROCESSED_FULL_PARQUET = config.DATA_PROCESSED / "f.parquet"
        config.UNSW_FILES = ["UNSW-NB15_1.csv"]

        # Reload modules so they see the patched paths
        from grammar import data_loader
        from experiments import e1_eda, e2_calibration, e3_figures
        for m in [data_loader, e1_eda, e2_calibration, e3_figures]:
            importlib.reload(m)

        # Step 1: EDA
        print("\n[1/3] EDA ...")
        eda_path = e1_eda.main()
        assert eda_path.exists(), f"EDA output missing: {eda_path}"
        print(f"      OK — {eda_path}")

        # Step 2: calibration
        print("\n[2/3] Calibration ...")
        cal_path = e2_calibration.main()
        assert cal_path.exists(), f"calibration output missing: {cal_path}"
        cal = pd.read_csv(cal_path)
        n_constraints = len(cal[cal["constraint_id"] != "OVERALL_ACTIVE_MEAN"])
        assert n_constraints == 12, f"expected 12 constraints, got {n_constraints}"
        print(f"      OK — {cal_path}  ({n_constraints} constraints + summary)")

        # Step 3: figures
        print("\n[3/3] Figures ...")
        fig_paths = e3_figures.main()
        for fp in fig_paths:
            assert fp.exists() and fp.stat().st_size > 0, f"figure missing or empty: {fp}"
        print(f"      OK — {len(fig_paths)} figures generated")

        print("\n" + "=" * 70)
        print("END-TO-END INTEGRATION TEST: ALL PASS")
        print("=" * 70)
        return 0

    finally:
        # restore config
        for k, v in original.items():
            setattr(config, k, v)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
