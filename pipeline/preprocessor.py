"""
PGCAV Preprocessor
==================
Encodes raw UNSW-NB15 rows into a feature matrix suitable for classifier
training, while preserving enough state to INVERT the transformation later.

Why the inverse matters: adversarial generation (Phase 2C) perturbs vectors
in the scaled feature space. To apply the grammar to those vectors, we have
to map them back to the original feature space. The Preprocessor stores
everything needed for that round trip.

Design:
* Categorical columns (proto, service, state) -> one-hot
* Numeric columns -> RobustScaler (insensitive to outliers, important here
  because UNSW-NB15 has heavy-tailed traffic features)
* Order-preserving: feature names recorded in self.feature_names_

Train/test split follows the official UNSW-NB15 convention:
    files 1-3 = train, file 4 = test.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config import (
    DROP_COLS_FOR_TRAINING,
    CATEGORICAL_COLS,
    LABEL_COL,
    MODELS_DIR,
    PROCESSED_TRAIN_PARQUET,
    PROCESSED_TEST_PARQUET,
    VERBOSE,
)
from grammar.data_loader import load_unsw_train_test


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[preproc] {msg}", flush=True)


class Preprocessor:
    """Fit on training rows, transform any rows of the same schema.

    State after `fit`:
        self.categorical_levels_ : dict[col, list of seen values]
        self.numeric_cols_       : list of numeric column names (post-drop)
        self.scaler_             : sklearn RobustScaler
        self.feature_names_      : final ordered feature names in X
    """

    def __init__(self):
        self.categorical_levels_: dict[str, list[str]] = {}
        self.numeric_cols_: list[str] = []
        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: list[str] = []
        self._fitted: bool = False

    # ── helpers ───────────────────────────────────────────────────────
    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if LABEL_COL not in df.columns:
            raise KeyError(f"target column '{LABEL_COL}' missing from DataFrame")
        y = df[LABEL_COL].astype(int)
        drop = [c for c in DROP_COLS_FOR_TRAINING if c in df.columns]
        X = df.drop(columns=drop, errors="ignore").copy()
        return X, y

    def _onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns using the fitted level sets.

        Unseen categories collapse to the existing 'other' bucket so the
        column count is stable between train and test.
        """
        out = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in out.columns:
                continue
            levels = self.categorical_levels_.get(col, [])
            # coerce unseen values to 'other'
            col_str = out[col].astype(str).str.lower()
            col_str = col_str.where(col_str.isin(levels), other="other")
            for lvl in levels:
                out[f"{col}__{lvl}"] = (col_str == lvl).astype(np.uint8)
            out = out.drop(columns=[col])
        return out

    # ── public API ────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X, _ = self._split_xy(df)

        # record categorical level sets — keep top-K most frequent + 'other'
        for col in CATEGORICAL_COLS:
            if col not in X.columns:
                continue
            vc = X[col].astype(str).str.lower().value_counts()
            top = vc.head(20).index.tolist()    # cap cardinality
            if "other" not in top:
                top.append("other")
            self.categorical_levels_[col] = top

        # apply categorical encoding to determine numeric column set
        X_enc = self._onehot(X)

        # numeric columns are everything except the categorical encodings
        cat_encoded = [c for c in X_enc.columns
                       if any(c.startswith(f"{cat}__") for cat in CATEGORICAL_COLS)]
        self.numeric_cols_ = [c for c in X_enc.columns if c not in cat_encoded]

        # fit a scaler on the numeric portion only
        X_num = X_enc[self.numeric_cols_].apply(pd.to_numeric, errors="coerce").fillna(0)
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X_num.values)

        # final column ordering: numeric (scaled) first, then categorical
        self.feature_names_ = list(self.numeric_cols_) + cat_encoded
        self._fitted = True
        _log(f"fit complete  numeric={len(self.numeric_cols_)}  "
             f"categorical_dims={len(cat_encoded)}  total={len(self.feature_names_)}")
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) with X scaled and column-aligned to fit-time."""
        if not self._fitted:
            raise RuntimeError("call .fit() before .transform()")
        X, y = self._split_xy(df)
        X_enc = self._onehot(X)

        # ensure all expected columns exist (add zero columns for missing)
        for col in self.feature_names_:
            if col not in X_enc.columns:
                X_enc[col] = 0
        X_enc = X_enc[self.feature_names_]   # exact ordering

        X_num = X_enc[self.numeric_cols_].apply(pd.to_numeric, errors="coerce").fillna(0)
        cat_part = X_enc.drop(columns=self.numeric_cols_).values.astype(np.float32)
        num_scaled = self.scaler_.transform(X_num.values).astype(np.float32)
        X_full = np.concatenate([num_scaled, cat_part], axis=1)
        return X_full, y.values.astype(np.int64)

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def inverse_numeric(self, X_scaled: np.ndarray) -> pd.DataFrame:
        """Undo the scaling on the numeric portion of the matrix.

        Returns a DataFrame whose columns are the original (lowercase) feature
        names. Categorical columns are NOT reconstructed — for adversarial
        validation we typically keep them frozen anyway.
        """
        if not self._fitted:
            raise RuntimeError("call .fit() first")
        n_num = len(self.numeric_cols_)
        X_num = X_scaled[:, :n_num]
        X_inv = self.scaler_.inverse_transform(X_num)
        return pd.DataFrame(X_inv, columns=self.numeric_cols_)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        _log(f"saved preprocessor to {path}")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Preprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"loaded object is {type(obj)}, not Preprocessor")
        return obj


# ──────────────────────────────────────────────────────────────────────
# Convenience: build the official train/test split
# ──────────────────────────────────────────────────────────────────────

def build_train_test_split(
    use_cache: bool = True,
    save_parquet: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the UNSW-NB15 train/test split, auto-detecting format.

    Format A (4-CSV raw): files 1-3 = train, file 4 = test.
    Format B (preprocessed): use the published train/test files directly.

    Returns
    -------
    (train_df, test_df) with all columns including 'label'.
    """
    if use_cache and PROCESSED_TRAIN_PARQUET.exists() and PROCESSED_TEST_PARQUET.exists():
        _log("loading cached train/test splits")
        return (
            pd.read_parquet(PROCESSED_TRAIN_PARQUET),
            pd.read_parquet(PROCESSED_TEST_PARQUET),
        )

    train_df, test_df = load_unsw_train_test()

    if save_parquet:
        train_df.to_parquet(PROCESSED_TRAIN_PARQUET, index=False)
        test_df.to_parquet(PROCESSED_TEST_PARQUET, index=False)
        _log("cached train/test as Parquet")

    return train_df, test_df
