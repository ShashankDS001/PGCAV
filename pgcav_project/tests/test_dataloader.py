"""
Loader smoke test
=================
Generates a tiny synthetic UNSW-NB15-shaped CSV and verifies the loader:
  1. correctly applies the 49-column header
  2. cleans hex port values
  3. coerces label/attack_cat
  4. extracts the normal subset

This test does NOT touch the real UNSW-NB15 files.
"""

from __future__ import annotations
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

import config
from grammar.data_loader import _UNSW_COLUMNS_FALLBACK, _clean_dataframe


def _build_synthetic_row(label: int, attack_cat: str = "") -> list:
    """Row with 49 fields matching the canonical schema."""
    rng = np.random.default_rng(42)
    base = {
        "srcip": "192.168.1.1", "sport": "0x000b", "dstip": "10.0.0.1",
        "dsport": "80", "proto": "tcp", "state": "FIN", "dur": 1.2,
        "sbytes": 1500, "dbytes": 3000, "sttl": 64, "dttl": 128,
        "sloss": 0, "dloss": 0, "service": "http",
        "sload": 10000.0, "dload": 20000.0, "spkts": 10, "dpkts": 20,
        "swin": 8192, "dwin": 8192, "stcpb": 100, "dtcpb": 200,
        "smeansz": 150, "dmeansz": 150, "trans_depth": 1, "res_bdy_len": 0,
        "sjit": 0.001, "djit": 0.002, "stime": 1421927414, "ltime": 1421927416,
        "sintpkt": 0.1, "dintpkt": 0.1, "tcprtt": 0.030,
        "synack": 0.015, "ackdat": 0.015, "is_sm_ips_ports": 0,
        "ct_state_ttl": 1, "ct_flw_http_mthd": 0, "is_ftp_login": "",
        "ct_ftp_cmd": "", "ct_srv_src": 1, "ct_srv_dst": 1,
        "ct_dst_ltm": 1, "ct_src_ltm": 1, "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1,
        "attack_cat": attack_cat, "label": label,
    }
    return [base[c] for c in _UNSW_COLUMNS_FALLBACK]


def test_clean_dataframe():
    print("Testing _clean_dataframe...")
    df = pd.DataFrame(
        [_build_synthetic_row(0, ""), _build_synthetic_row(1, "Exploits")],
        columns=_UNSW_COLUMNS_FALLBACK,
    )
    cleaned = _clean_dataframe(df.copy())

    # hex port converted
    assert cleaned["sport"].iloc[0] == float(0x000b), \
        f"hex port not converted: got {cleaned['sport'].iloc[0]}"
    print("  PASS  hex port conversion")

    # label coerced to int
    assert cleaned["label"].dtype.kind in "iu", \
        f"label not coerced to int: dtype={cleaned['label'].dtype}"
    print("  PASS  label coerced to int")

    # empty attack_cat → 'normal'
    assert cleaned["attack_cat"].iloc[0] == "normal", \
        f"empty attack_cat not normalised: got {cleaned['attack_cat'].iloc[0]}"
    print("  PASS  attack_cat 'normal' normalisation")

    # lowercase preserved
    assert cleaned["attack_cat"].iloc[1] == "exploits", \
        f"attack_cat not lowercased: got {cleaned['attack_cat'].iloc[1]}"
    print("  PASS  attack_cat lowercasing")

    # ct_ftp_cmd / is_ftp_login coerced from empty string
    assert pd.api.types.is_numeric_dtype(cleaned["ct_ftp_cmd"]), \
        f"ct_ftp_cmd not numeric: dtype={cleaned['ct_ftp_cmd'].dtype}"
    print("  PASS  ct_ftp_cmd coerced")

    print("test_clean_dataframe: ALL PASS")
    return True


def test_full_loader_with_synthetic_csv():
    """End-to-end Format A: write fake 4-CSV layout, verify load + clean."""
    print("\nTesting full loader on synthetic Format A (4-CSV) layout...")

    tmp = Path(tempfile.mkdtemp(prefix="pgcav_test_A_"))
    try:
        # generate ALL 4 CSVs (auto-detector needs them all to choose format A)
        for fname in ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv",
                      "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]:
            rows = [_build_synthetic_row(0, "") for _ in range(30)]
            rows += [_build_synthetic_row(1, "DoS") for _ in range(10)]
            df = pd.DataFrame(rows)
            df.to_csv(tmp / fname, header=False, index=False)

        # monkey-patch config paths
        original = {
            "DATA_RAW": config.DATA_RAW,
            "DATA_PROCESSED": config.DATA_PROCESSED,
            "PROCESSED_NORMAL_PARQUET": config.PROCESSED_NORMAL_PARQUET,
            "PROCESSED_FULL_PARQUET": config.PROCESSED_FULL_PARQUET,
        }
        config.DATA_RAW = tmp
        config.DATA_PROCESSED = tmp / "processed"
        config.DATA_PROCESSED.mkdir(exist_ok=True)
        config.PROCESSED_NORMAL_PARQUET = config.DATA_PROCESSED / "normal.parquet"
        config.PROCESSED_FULL_PARQUET = config.DATA_PROCESSED / "full.parquet"

        import importlib
        from grammar import data_loader
        importlib.reload(data_loader)

        fmt = data_loader.detect_format()
        assert fmt == "A", f"expected format A, got {fmt}"
        print("  PASS  format A detected")

        loaded = data_loader.load_unsw_raw(use_cache=False)
        assert loaded.shape == (160, 49), f"unexpected shape {loaded.shape}"
        print(f"  PASS  loaded shape: {loaded.shape}")

        normal = loaded[loaded["label"] == 0]
        assert len(normal) == 120, f"expected 120 normal rows, got {len(normal)}"
        print(f"  PASS  normal count: {len(normal)}")

        assert loaded["sport"].dtype.kind == "f", \
            f"sport not float after hex coerce: {loaded['sport'].dtype}"
        print("  PASS  sport column numeric")

    finally:
        for k, v in original.items():
            setattr(config, k, v)
        shutil.rmtree(tmp, ignore_errors=True)

    print("test_full_loader_with_synthetic_csv: ALL PASS")
    return True


def test_format_B_parquet_loader():
    """Format B: write fake training-set/testing-set parquet, verify load + alias."""
    print("\nTesting full loader on synthetic Format B (parquet) layout...")

    tmp = Path(tempfile.mkdtemp(prefix="pgcav_test_B_"))
    try:
        # build a format-B-shaped DataFrame with the renamed columns
        # 45 cols: format-A 49 minus {srcip, sport, dstip, dsport, stime, ltime}
        # plus {id, rate}, with smean/dmean/response_body_len/sinpkt/dinpkt names
        rng = np.random.default_rng(0)

        def build_b(n_normal, n_attack):
            n = n_normal + n_attack
            return pd.DataFrame({
                "id": np.arange(n),
                "dur": rng.uniform(0, 10, n),
                "proto": rng.choice(["tcp", "udp", "icmp"], n),
                "service": rng.choice(["http", "dns", "-"], n),
                "state": rng.choice(["FIN", "INT", "CON"], n),
                "spkts": rng.integers(1, 50, n),
                "dpkts": rng.integers(1, 50, n),
                "sbytes": rng.integers(100, 5000, n),
                "dbytes": rng.integers(100, 5000, n),
                "rate": rng.uniform(0, 1000, n),
                "sttl": rng.integers(32, 255, n),
                "dttl": rng.integers(32, 255, n),
                "sload": rng.uniform(0, 1e6, n),
                "dload": rng.uniform(0, 1e6, n),
                "sloss": rng.integers(0, 3, n),
                "dloss": rng.integers(0, 3, n),
                "sinpkt": rng.uniform(0, 0.1, n),     # ← format B name
                "dinpkt": rng.uniform(0, 0.1, n),     # ← format B name
                "sjit": rng.uniform(0, 0.05, n),
                "djit": rng.uniform(0, 0.05, n),
                "swin": rng.integers(0, 65535, n),
                "stcpb": rng.integers(0, 4_000_000_000, n),
                "dtcpb": rng.integers(0, 4_000_000_000, n),
                "dwin": rng.integers(0, 65535, n),
                "tcprtt": rng.uniform(0, 0.1, n),
                "synack": rng.uniform(0, 0.05, n),
                "ackdat": rng.uniform(0, 0.05, n),
                "smean": rng.integers(40, 1500, n),         # ← format B name
                "dmean": rng.integers(40, 1500, n),         # ← format B name
                "trans_depth": rng.integers(0, 5, n),
                "response_body_len": rng.integers(0, 1000, n),  # ← format B name
                "ct_srv_src": rng.integers(0, 50, n),
                "ct_state_ttl": rng.integers(0, 10, n),
                "ct_dst_ltm": rng.integers(0, 20, n),
                "ct_src_dport_ltm": rng.integers(0, 20, n),
                "ct_dst_sport_ltm": rng.integers(0, 20, n),
                "ct_dst_src_ltm": rng.integers(0, 20, n),
                "is_ftp_login": rng.integers(0, 2, n),
                "ct_ftp_cmd": rng.integers(0, 5, n),
                "ct_flw_http_mthd": rng.integers(0, 5, n),
                "ct_src_ltm": rng.integers(0, 20, n),
                "ct_srv_dst": rng.integers(0, 50, n),
                "is_sm_ips_ports": rng.integers(0, 2, n),
                "attack_cat": ["Normal"] * n_normal + ["DoS"] * n_attack,
                "label": [0] * n_normal + [1] * n_attack,
            })

        train = build_b(60, 40)
        test = build_b(30, 20)
        train.to_parquet(tmp / "UNSW_NB15_training-set.parquet", index=False)
        test.to_parquet(tmp / "UNSW_NB15_testing-set.parquet", index=False)

        original = {
            "DATA_RAW": config.DATA_RAW,
            "DATA_PROCESSED": config.DATA_PROCESSED,
            "PROCESSED_NORMAL_PARQUET": config.PROCESSED_NORMAL_PARQUET,
            "PROCESSED_FULL_PARQUET": config.PROCESSED_FULL_PARQUET,
        }
        config.DATA_RAW = tmp
        config.DATA_PROCESSED = tmp / "processed"
        config.DATA_PROCESSED.mkdir(exist_ok=True)
        config.PROCESSED_NORMAL_PARQUET = config.DATA_PROCESSED / "normal.parquet"
        config.PROCESSED_FULL_PARQUET = config.DATA_PROCESSED / "full.parquet"

        import importlib
        from grammar import data_loader
        importlib.reload(data_loader)

        fmt = data_loader.detect_format()
        assert fmt == "B", f"expected format B, got {fmt}"
        print("  PASS  format B detected")

        loaded = data_loader.load_unsw_raw(use_cache=False)
        assert loaded.shape[0] == 150, f"expected 150 rows total, got {loaded.shape[0]}"
        print(f"  PASS  loaded shape: {loaded.shape}")

        # Column aliasing: smean → smeansz, dmean → dmeansz, etc.
        assert "smeansz" in loaded.columns and "smean" not in loaded.columns
        print("  PASS  smean -> smeansz aliased")
        assert "dmeansz" in loaded.columns and "dmean" not in loaded.columns
        print("  PASS  dmean -> dmeansz aliased")
        assert "res_bdy_len" in loaded.columns
        print("  PASS  response_body_len -> res_bdy_len aliased")
        assert "sintpkt" in loaded.columns
        print("  PASS  sinpkt -> sintpkt aliased")
        assert "id" not in loaded.columns
        print("  PASS  id column dropped")

        normal = loaded[loaded["label"] == 0]
        assert len(normal) == 90, f"expected 90 normal rows, got {len(normal)}"
        print(f"  PASS  normal count: {len(normal)}")

        # Train/test split using the published files
        train_df, test_df = data_loader.load_unsw_train_test()
        assert len(train_df) == 100 and len(test_df) == 50, \
            f"expected 100/50, got {len(train_df)}/{len(test_df)}"
        print(f"  PASS  train/test split: {len(train_df)}/{len(test_df)}")

    finally:
        for k, v in original.items():
            setattr(config, k, v)
        shutil.rmtree(tmp, ignore_errors=True)

    print("test_format_B_parquet_loader: ALL PASS")
    return True


if __name__ == "__main__":
    ok = True
    try:
        ok &= test_clean_dataframe()
        ok &= test_full_loader_with_synthetic_csv()
        ok &= test_format_B_parquet_loader()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        ok = False
    print(f"\nOVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)
