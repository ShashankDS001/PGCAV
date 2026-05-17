"""
Pipeline integration test
=========================
Tests preprocessor + classifier training on synthetic data — verifies
the Phase 2A code runs end-to-end before you commit to the ~10 min
training run on the real UNSW-NB15 dataset.

Run:
    python tests/test_pipeline.py
"""

from __future__ import annotations
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pipeline.preprocessor import Preprocessor
from pipeline.classifiers import (
    train_rf, train_mlp, save_model, load_model, evaluate_model, metrics_table,
)
from grammar.data_loader import _UNSW_COLUMNS_FALLBACK


def build_synthetic_dataset(n_normal: int = 2000, n_attack: int = 500,
                            seed: int = 42) -> pd.DataFrame:
    """Synthetic UNSW-NB15-shaped data with separable classes.

    We inject a class-correlated signal into a couple of features so the
    classifiers can actually learn something — without that they'd hit
    random accuracy and the test would be uninformative.
    """
    rng = np.random.default_rng(seed)
    n = n_normal + n_attack
    proto = rng.choice(["tcp", "udp", "icmp"], size=n, p=[0.7, 0.2, 0.1])
    service = rng.choice(["http", "dns", "smtp", "ftp", "-"], size=n)
    state = rng.choice(["FIN", "INT", "CON"], size=n)
    label = np.concatenate([np.zeros(n_normal), np.ones(n_attack)]).astype(int)

    # Inject class signal: attack rows have systematically different sttl/dur
    sttl_normal = rng.integers(60, 80, size=n_normal)
    sttl_attack = rng.integers(30, 50, size=n_attack)
    sttl = np.concatenate([sttl_normal, sttl_attack])

    dur_normal = rng.uniform(0, 1, size=n_normal)
    dur_attack = rng.uniform(2, 10, size=n_attack)
    dur = np.concatenate([dur_normal, dur_attack])

    df = pd.DataFrame({
        "srcip": ["10.0.0.1"] * n,
        "sport": rng.integers(1024, 65535, size=n),
        "dstip": ["10.0.0.2"] * n,
        "dsport": rng.integers(1, 65535, size=n),
        "proto": proto, "state": state, "dur": dur,
        "sbytes": rng.integers(100, 10000, size=n),
        "dbytes": rng.integers(100, 10000, size=n),
        "sttl": sttl,
        "dttl": rng.integers(32, 255, size=n),
        "sloss": rng.integers(0, 3, size=n), "dloss": rng.integers(0, 3, size=n),
        "service": service,
        "sload": rng.uniform(0, 1e6, size=n), "dload": rng.uniform(0, 1e6, size=n),
        "spkts": rng.integers(1, 100, size=n), "dpkts": rng.integers(1, 100, size=n),
        "swin": rng.integers(0, 65535, size=n), "dwin": rng.integers(0, 65535, size=n),
        "stcpb": rng.integers(0, 4_000_000_000, size=n),
        "dtcpb": rng.integers(0, 4_000_000_000, size=n),
        "smeansz": rng.integers(40, 1500, size=n),
        "dmeansz": rng.integers(40, 1500, size=n),
        "trans_depth": rng.integers(0, 5, size=n),
        "res_bdy_len": rng.integers(0, 1000, size=n),
        "sjit": rng.uniform(0, 0.05, size=n), "djit": rng.uniform(0, 0.05, size=n),
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
        "attack_cat": ["normal"] * n_normal + ["DoS"] * n_attack,
        "label": label,
    })
    # shuffle so train/test split below has both classes
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df[_UNSW_COLUMNS_FALLBACK]
    return df


def _check(label: str, cond: bool, detail: str = "") -> bool:
    flag = "PASS" if cond else "FAIL"
    extra = f"  -- {detail}" if detail else ""
    print(f"  {flag}  {label}{extra}")
    return cond


def main() -> int:
    print("Running pipeline integration tests on synthetic data...")
    print("-" * 70)
    ok = True

    # Build synthetic train / test
    train_df = build_synthetic_dataset(n_normal=2000, n_attack=500, seed=42)
    test_df = build_synthetic_dataset(n_normal=600, n_attack=200, seed=43)

    # ── Preprocessor ────────────────────────────────────────────────
    pre = Preprocessor()
    X_train, y_train = pre.fit_transform(train_df)
    X_test, y_test = pre.transform(test_df)

    ok &= _check("preprocessor produces 2-D float matrix",
                 X_train.ndim == 2 and X_train.dtype == np.float32,
                 f"shape={X_train.shape} dtype={X_train.dtype}")
    ok &= _check("train and test have same feature dimensionality",
                 X_train.shape[1] == X_test.shape[1],
                 f"train d={X_train.shape[1]} test d={X_test.shape[1]}")
    ok &= _check("y_train has both classes",
                 set(np.unique(y_train).tolist()) == {0, 1})
    ok &= _check("preprocessor stored feature_names_",
                 len(pre.feature_names_) == X_train.shape[1])

    # ── Inverse transform ───────────────────────────────────────────
    inv = pre.inverse_numeric(X_train[:5])
    ok &= _check("inverse_numeric returns DataFrame with numeric cols",
                 set(inv.columns) == set(pre.numeric_cols_))

    # ── Save / load preprocessor ────────────────────────────────────
    tmp = Path(tempfile.mkdtemp(prefix="pgcav_pipe_"))
    try:
        pre_path = pre.save(tmp / "preprocessor.pkl")
        pre2 = Preprocessor.load(pre_path)
        X_again, _ = pre2.transform(test_df)
        ok &= _check("preprocessor saves and reloads identically",
                     np.allclose(X_test, X_again))

        # ── Train RF and MLP ────────────────────────────────────────
        rf = train_rf(X_train, y_train)
        mlp = train_mlp(X_train, y_train, max_iter=20)

        # Save and reload
        save_model(rf, "rf_test", models_dir=tmp)
        save_model(mlp, "mlp_test", models_dir=tmp)
        rf2 = load_model("rf_test", models_dir=tmp)
        mlp2 = load_model("mlp_test", models_dir=tmp)

        ok &= _check("RF reloads and predicts identically",
                     np.array_equal(rf.predict(X_test), rf2.predict(X_test)))
        ok &= _check("MLP reloads and predicts identically",
                     np.array_equal(mlp.predict(X_test), mlp2.predict(X_test)))

        # ── Evaluate ────────────────────────────────────────────────
        rf_metrics = evaluate_model(rf, X_test, y_test, name="RF_test")
        mlp_metrics = evaluate_model(mlp, X_test, y_test, name="MLP_test")
        print(f"\n  RF  test accuracy: {rf_metrics['accuracy']:.4f}   "
              f"f1: {rf_metrics['f1']:.4f}   auc: {rf_metrics['roc_auc']:.4f}")
        print(f"  MLP test accuracy: {mlp_metrics['accuracy']:.4f}   "
              f"f1: {mlp_metrics['f1']:.4f}   auc: {mlp_metrics['roc_auc']:.4f}\n")

        # The injected class signal is strong, so both should be well above chance
        ok &= _check("RF beats chance (acc > 0.70)",
                     rf_metrics["accuracy"] > 0.70,
                     f"acc={rf_metrics['accuracy']:.4f}")
        ok &= _check("MLP beats chance (acc > 0.70)",
                     mlp_metrics["accuracy"] > 0.70,
                     f"acc={mlp_metrics['accuracy']:.4f}")

        # Metrics table assembly
        table = metrics_table([rf_metrics, mlp_metrics])
        ok &= _check("metrics_table builds a DataFrame",
                     isinstance(table, pd.DataFrame) and len(table) == 2)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 70)
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
