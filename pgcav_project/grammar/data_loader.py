"""
UNSW-NB15 Data Loader  (auto-detecting)
=======================================
Handles the TWO published distributions of UNSW-NB15.

Format A — original 4-CSV raw capture (~2.54M rows, 49 features, no header)
    UNSW-NB15_1.csv ... UNSW-NB15_4.csv
    NUSW-NB15_features.csv

Format B — preprocessed train/test split (~257k rows, 45 features, with header)
    UNSW_NB15_training-set.{parquet,csv}
    UNSW_NB15_testing-set.{parquet,csv}

The loader auto-detects which is present in data/raw/ and routes accordingly.
For format B, it applies a column-alias map so downstream code (constraints,
preprocessor) sees the same names as format A. The renames applied:

    smean              -> smeansz       (used by C10)
    dmean              -> dmeansz
    response_body_len  -> res_bdy_len
    sinpkt             -> sintpkt
    dinpkt             -> dintpkt

Common ingestion fixes applied to both formats:
* lowercase all column names
* coerce sport/dsport hex strings to floats (format A only)
* coerce ct_ftp_cmd, is_ftp_login empty strings to numeric
* normalise attack_cat (whitespace, casing, empty → 'normal')
* coerce label to int 0/1
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import sys

import pandas as pd
import numpy as np

from config import (
    DATA_RAW,
    DATA_PROCESSED,
    UNSW_FILES,
    UNSW_TRAIN_FILES,
    UNSW_TEST_FILE,
    UNSW_FEATURES_FILE,
    UNSW_PARQUET_TRAIN,
    UNSW_PARQUET_TEST,
    UNSW_PREPROC_CSV_TRAIN,
    UNSW_PREPROC_CSV_TEST,
    PROCESSED_NORMAL_PARQUET,
    PROCESSED_FULL_PARQUET,
    LABEL_COL,
    ATTACK_COL,
    VERBOSE,
)


# ── Canonical 49 columns for format A ─────────────────────────────────
_UNSW_COLUMNS_FALLBACK = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
    "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

# ── Format B → format A column alias map ─────────────────────────────
_PREPROC_RENAMES = {
    "smean": "smeansz",
    "dmean": "dmeansz",
    "response_body_len": "res_bdy_len",
    "sinpkt": "sintpkt",
    "dinpkt": "dintpkt",
}


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[loader] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Common cleanup applied to both formats
# ──────────────────────────────────────────────────────────────────────

def _coerce_port(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "" or s == "-":
        return np.nan
    try:
        if s.lower().startswith("0x"):
            return float(int(s, 16))
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all the UNSW-NB15-specific cleanups."""
    # Coerce port columns from possibly-hex strings (format A only)
    for col in ("sport", "dsport"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(_coerce_port)

    # Numerical columns that occasionally have empty strings
    for col in ("ct_ftp_cmd", "is_ftp_login"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Lowercase categorical strings
    for col in ("proto", "service", "state", ATTACK_COL):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # attack_cat: empty strings → 'normal'
    if ATTACK_COL in df.columns:
        df[ATTACK_COL] = df[ATTACK_COL].replace(
            {"": "normal", "nan": "normal", "none": "normal"}
        )

    # label must be 0/1 int
    if LABEL_COL in df.columns:
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    return df


def _apply_preproc_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Map format-B column names to format-A names."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    renames = {old: new for old, new in _PREPROC_RENAMES.items()
               if old in df.columns and new not in df.columns}
    if renames:
        df = df.rename(columns=renames)
        _log(f"applied column aliases: {renames}")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df


# ──────────────────────────────────────────────────────────────────────
# Format A: 4-CSV raw capture
# ──────────────────────────────────────────────────────────────────────

def get_unsw_columns(features_path: Optional[Path] = None) -> list[str]:
    """Return the 49 UNSW-NB15 column names from the features CSV."""
    if features_path is None:
        features_path = DATA_RAW / UNSW_FEATURES_FILE

    if not features_path.exists():
        _log(f"features file not found at {features_path}, using fallback")
        return list(_UNSW_COLUMNS_FALLBACK)

    try:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                feats = pd.read_csv(features_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError("could not decode features CSV")

        name_col = None
        for cand in ("Name", "name", "NAME", "feature", "Feature"):
            if cand in feats.columns:
                name_col = cand
                break
        if name_col is None:
            return list(_UNSW_COLUMNS_FALLBACK)

        names = feats[name_col].astype(str).str.strip().str.lower().tolist()
        if len(names) != 49:
            return list(_UNSW_COLUMNS_FALLBACK)
        return names
    except Exception as e:
        _log(f"error reading features file: {e} — using fallback")
        return list(_UNSW_COLUMNS_FALLBACK)


def _load_raw_csv_set(files: list[str]) -> pd.DataFrame:
    """Concatenate the format-A CSV files into one DataFrame."""
    columns = get_unsw_columns()
    frames = []
    for fname in files:
        fpath = DATA_RAW / fname
        if not fpath.exists():
            raise FileNotFoundError(f"missing CSV: {fpath}")
        _log(f"reading {fname} ...")
        df = pd.read_csv(
            fpath, header=None, names=columns,
            low_memory=False, encoding="latin-1", on_bad_lines="skip",
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Format B: preprocessed train/test parquet (or CSV)
# ──────────────────────────────────────────────────────────────────────

def _load_preprocessed_file(path: Path) -> pd.DataFrame:
    """Load a single format-B file (parquet or CSV)."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df = _apply_preproc_renames(df)
    return df


# ──────────────────────────────────────────────────────────────────────
# Auto-detection
# ──────────────────────────────────────────────────────────────────────

def _detect_format() -> str:
    """Return 'A' if 4-CSV raw is present, 'B' if preprocessed train/test, else raise."""
    a_present = all((DATA_RAW / f).exists() for f in UNSW_FILES)

    b_parquet_train = DATA_RAW / UNSW_PARQUET_TRAIN
    b_parquet_test = DATA_RAW / UNSW_PARQUET_TEST
    b_csv_train = DATA_RAW / UNSW_PREPROC_CSV_TRAIN
    b_csv_test = DATA_RAW / UNSW_PREPROC_CSV_TEST

    b_parquet_present = b_parquet_train.exists() and b_parquet_test.exists()
    b_csv_present = b_csv_train.exists() and b_csv_test.exists()
    b_present = b_parquet_present or b_csv_present

    if a_present:
        return "A"
    if b_present:
        return "B"

    raise FileNotFoundError(
        f"\n  Could not find UNSW-NB15 dataset files in {DATA_RAW}/\n"
        f"  Looking for ONE of these layouts:\n"
        f"    Format A (raw, ~2.54M rows):\n"
        f"      {UNSW_FILES[0]}, {UNSW_FILES[1]}, {UNSW_FILES[2]}, {UNSW_FILES[3]}\n"
        f"      {UNSW_FEATURES_FILE}\n"
        f"    Format B (preprocessed, ~257k rows):\n"
        f"      {UNSW_PARQUET_TRAIN}\n"
        f"      {UNSW_PARQUET_TEST}\n"
        f"    or the CSV equivalents.\n"
        f"  Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
    )


def detect_format() -> str:
    """Public detector — returns 'A' or 'B'."""
    return _detect_format()


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def load_unsw_raw(
    files: Optional[list[str]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load the full UNSW-NB15 dataset, auto-detecting format.

    Format A: optionally restrict to a subset of CSV files via `files`.
    Format B: `files` is ignored and both train+test are concatenated.

    Returns a single DataFrame with format-A column names (post-aliasing).
    """
    if use_cache and PROCESSED_FULL_PARQUET.exists():
        _log(f"loading cache: {PROCESSED_FULL_PARQUET}")
        return pd.read_parquet(PROCESSED_FULL_PARQUET)

    fmt = _detect_format()
    _log(f"detected format: {fmt}")

    if fmt == "A":
        if files is None:
            files = UNSW_FILES
        df_full = _load_raw_csv_set(files)
    else:
        # Format B: concatenate train + test for the "full" view
        train_path = DATA_RAW / UNSW_PARQUET_TRAIN
        if not train_path.exists():
            train_path = DATA_RAW / UNSW_PREPROC_CSV_TRAIN
        test_path = DATA_RAW / UNSW_PARQUET_TEST
        if not test_path.exists():
            test_path = DATA_RAW / UNSW_PREPROC_CSV_TEST
        _log(f"reading {train_path.name} ...")
        df_train = _load_preprocessed_file(train_path)
        _log(f"reading {test_path.name} ...")
        df_test = _load_preprocessed_file(test_path)
        df_full = pd.concat([df_train, df_test], ignore_index=True)

    _log(f"concatenated shape: {df_full.shape}")
    df_full = _clean_dataframe(df_full)

    if use_cache:
        _log(f"caching to {PROCESSED_FULL_PARQUET}")
        df_full.to_parquet(PROCESSED_FULL_PARQUET, index=False)

    return df_full


def load_unsw_normal(use_cache: bool = True) -> pd.DataFrame:
    """Return only the clean (label=0) traffic subset."""
    if use_cache and PROCESSED_NORMAL_PARQUET.exists():
        _log(f"loading cache: {PROCESSED_NORMAL_PARQUET}")
        return pd.read_parquet(PROCESSED_NORMAL_PARQUET)

    df = load_unsw_raw(use_cache=use_cache)
    if LABEL_COL not in df.columns:
        raise RuntimeError(
            f"'{LABEL_COL}' column missing. Columns present: {df.columns.tolist()}"
        )
    normal = df[df[LABEL_COL] == 0].copy().reset_index(drop=True)
    _log(f"normal rows: {len(normal):,}")

    if use_cache:
        normal.to_parquet(PROCESSED_NORMAL_PARQUET, index=False)

    return normal


def load_unsw_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) respecting the official split.

    Format A: files 1-3 = train, file 4 = test.
    Format B: use the published training-set / testing-set files directly.
    """
    fmt = _detect_format()
    _log(f"detected format: {fmt}  (for train/test split)")

    if fmt == "A":
        train_df = _load_raw_csv_set(UNSW_TRAIN_FILES)
        test_df = _load_raw_csv_set(UNSW_TEST_FILE)
    else:
        train_path = DATA_RAW / UNSW_PARQUET_TRAIN
        if not train_path.exists():
            train_path = DATA_RAW / UNSW_PREPROC_CSV_TRAIN
        test_path = DATA_RAW / UNSW_PARQUET_TEST
        if not test_path.exists():
            test_path = DATA_RAW / UNSW_PREPROC_CSV_TEST
        train_df = _load_preprocessed_file(train_path)
        test_df = _load_preprocessed_file(test_path)

    train_df = _clean_dataframe(train_df)
    test_df = _clean_dataframe(test_df)
    _log(f"train={train_df.shape}  test={test_df.shape}")
    return train_df, test_df


def quick_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"DataFrame summary  shape={df.shape}")
    print("=" * 60)
    if LABEL_COL in df.columns:
        vc = df[LABEL_COL].value_counts()
        print(f"  label=0 (normal): {vc.get(0, 0):>10,}")
        print(f"  label=1 (attack): {vc.get(1, 0):>10,}")
    if ATTACK_COL in df.columns:
        print(f"\n  attack categories:")
        for cat, n in df[ATTACK_COL].value_counts().head(15).items():
            print(f"    {cat:<20} {n:>10,}")
    print()


if __name__ == "__main__":
    try:
        fmt = detect_format()
        print(f"Detected format: {fmt}")
        df = load_unsw_raw()
        quick_summary(df)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
