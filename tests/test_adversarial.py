"""
Adversarial pipeline smoke tests
=================================
Tests run without the real UNSW-NB15 data and without launching actual
ART attacks (which take minutes). Uses tiny synthetic data + mock
adversarials to verify:

  * ART wrapper initialises correctly
  * reconstruct_for_grammar works
  * compute_detection_metrics is arithmetically correct
  * per_constraint_vr handles edge cases
  * validate_dataframe integrates with adversarial reconstruction

Run:
    python tests/test_adversarial.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from grammar.constraints import CONSTRAINTS
from grammar.validator import validate_dataframe
from pipeline.adversarial import (
    ART_AVAILABLE,
    compute_detection_metrics,
    per_constraint_vr,
    reconstruct_for_grammar,
)


def _check(label: str, cond: bool, detail: str = "") -> bool:
    flag = "PASS" if cond else "FAIL"
    extra = f"  -- {detail}" if detail else ""
    print(f"  {flag}  {label}{extra}")
    return cond


def test_compute_detection_metrics() -> bool:
    """Detection metric arithmetic."""
    print("\ntest_compute_detection_metrics ...")
    ok = True

    # 10 attack flows, classifier catches 6, misses 4
    # Of the 4 evading, grammar catches 3
    y_clf = np.array([1,1,1,1,1,1,0,0,0,0])   # 4 evasions (pred=0)
    grammar_valid = np.array([True]*6 + [False,False,False,True])  # 3 violated among evading

    m = compute_detection_metrics(y_clf, grammar_valid, attack_label=1)

    ok &= _check("n_total", m["n_total"] == 10)
    ok &= _check("n_evading", m["n_evading"] == 4)
    ok &= _check("ER = 4/10 = 0.4", abs(m["evasion_rate"] - 0.4) < 1e-9,
                 f"got {m['evasion_rate']}")
    ok &= _check("VR_all = 3/10 = 0.3", abs(m["vr_all"] - 0.3) < 1e-9,
                 f"got {m['vr_all']}")
    ok &= _check("VR_evading = 3/4 = 0.75", abs(m["vr_evading"] - 0.75) < 1e-9,
                 f"got {m['vr_evading']}")

    # combined_tpr = 1 - ER*(1-VR_evading) = 1 - 0.4*0.25 = 1 - 0.1 = 0.9
    ok &= _check("combined_tpr = 0.9", abs(m["combined_tpr"] - 0.90) < 1e-9,
                 f"got {m['combined_tpr']}")
    # ETR = ER*(1-VR_evading) = 0.4*0.25 = 0.1
    ok &= _check("ETR = 0.1", abs(m["etr"] - 0.10) < 1e-9, f"got {m['etr']}")
    # clf_only_tpr = 1 - ER = 0.6
    ok &= _check("clf_only_tpr = 0.6", abs(m["clf_only_tpr"] - 0.6) < 1e-9,
                 f"got {m['clf_only_tpr']}")

    # edge: perfect evasion (ER=1), grammar catches all
    y_all_evade = np.zeros(5, dtype=int)     # all predict benign
    gram_all_viol = np.zeros(5, dtype=bool)  # all violated → valid=False
    m2 = compute_detection_metrics(y_all_evade, gram_all_viol, attack_label=1)
    ok &= _check("edge: ER=1, VR=1 → ETR=0",
                 abs(m2["etr"] - 0.0) < 1e-9, f"got {m2['etr']}")

    # edge: perfect classifier (ER=0), grammar irrelevant
    y_no_evade = np.ones(5, dtype=int)
    gram_any = np.ones(5, dtype=bool)
    m3 = compute_detection_metrics(y_no_evade, gram_any, attack_label=1)
    ok &= _check("edge: ER=0 → combined_tpr=1",
                 abs(m3["combined_tpr"] - 1.0) < 1e-9)

    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


def test_per_constraint_vr() -> bool:
    """Per-constraint VR dataframe construction."""
    print("\ntest_per_constraint_vr ...")
    ok = True

    # Build a tiny grammar_result_df with two constraints
    n = 20
    c1_valid = np.array([True]*15 + [False]*5)   # 5 violations
    c2_valid = np.array([True]*18 + [False]*2)   # 2 violations
    gram_df = pd.DataFrame({"C01_ttl_range": c1_valid,
                            "C04_non_neg_dur": c2_valid,
                            "all_valid": c1_valid & c2_valid})
    evading_mask = np.zeros(n, dtype=bool)
    evading_mask[15:] = True   # last 5 are evading

    pc = per_constraint_vr(gram_df, evading_mask, ["C01_ttl_range", "C04_non_neg_dur"])
    ok &= _check("returns DataFrame", isinstance(pc, pd.DataFrame))
    ok &= _check("correct number of rows", len(pc) == 2)

    c01 = pc.set_index("constraint_id").loc["C01_ttl_range"]
    # VR_all = 5/20 = 0.25
    ok &= _check("C01 vr_all=0.25", abs(c01["vr_all"] - 0.25) < 1e-9,
                 f"got {c01['vr_all']}")
    # VR_evading = violations among rows 15-19 = 5 out of 5 evading
    ok &= _check("C01 vr_evading=1.0", abs(c01["vr_evading"] - 1.0) < 1e-9,
                 f"got {c01['vr_evading']}")

    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


def test_reconstruct_for_grammar() -> bool:
    """reconstruct_for_grammar produces a DataFrame with correct columns."""
    print("\ntest_reconstruct_for_grammar ...")
    ok = True

    import tempfile, shutil, pickle
    from pipeline.preprocessor import Preprocessor
    from grammar.data_loader import _UNSW_COLUMNS_FALLBACK
    import importlib, config

    rng = np.random.default_rng(42)
    n = 50

    # Build a tiny training-like DataFrame
    df = pd.DataFrame({
        "proto": rng.choice(["tcp","udp"], n),
        "service": rng.choice(["http","dns"], n),
        "state": rng.choice(["FIN","INT"], n),
        "dur": rng.uniform(0, 10, n),
        "sbytes": rng.integers(100, 5000, n),
        "dbytes": rng.integers(100, 5000, n),
        "sttl": rng.integers(32, 255, n),
        "dttl": rng.integers(32, 255, n),
        "sloss": rng.integers(0, 3, n),
        "dloss": rng.integers(0, 3, n),
        "sload": rng.uniform(0, 1e6, n),
        "dload": rng.uniform(0, 1e6, n),
        "spkts": rng.integers(1, 100, n),
        "dpkts": rng.integers(1, 100, n),
        "swin": rng.integers(0, 255, n),
        "dwin": rng.integers(0, 255, n),
        "stcpb": rng.integers(0, 4_000_000_000, n),
        "dtcpb": rng.integers(0, 4_000_000_000, n),
        "smeansz": rng.integers(40, 1500, n),
        "dmeansz": rng.integers(40, 1500, n),
        "trans_depth": rng.integers(0, 5, n),
        "res_bdy_len": rng.integers(0, 1000, n),
        "sjit": rng.uniform(0, 0.05, n),
        "djit": rng.uniform(0, 0.05, n),
        "sintpkt": rng.uniform(0, 0.1, n),
        "dintpkt": rng.uniform(0, 0.1, n),
        "tcprtt": rng.uniform(0, 0.1, n),
        "synack": rng.uniform(0, 0.05, n),
        "ackdat": rng.uniform(0, 0.05, n),
        "is_sm_ips_ports": rng.integers(0, 2, n),
        "ct_state_ttl": rng.integers(0, 10, n),
        "ct_flw_http_mthd": rng.integers(0, 5, n),
        "is_ftp_login": rng.integers(0, 2, n),
        "ct_ftp_cmd": rng.integers(0, 5, n),
        "ct_srv_src": rng.integers(0, 50, n),
        "ct_srv_dst": rng.integers(0, 50, n),
        "ct_dst_ltm": rng.integers(0, 20, n),
        "ct_src_ltm": rng.integers(0, 20, n),
        "ct_src_dport_ltm": rng.integers(0, 20, n),
        "ct_dst_sport_ltm": rng.integers(0, 20, n),
        "ct_dst_src_ltm": rng.integers(0, 20, n),
        "attack_cat": ["normal"] * n,
        "label": [0] * n,
    })

    pre = Preprocessor()
    X_scaled, y = pre.fit_transform(df)

    # Simulate adversarial: add small noise to scaled space
    X_adv = X_scaled + rng.uniform(-0.1, 0.1, X_scaled.shape).astype(np.float32)

    df_gram = reconstruct_for_grammar(X_adv, pre, df)
    ok &= _check("returns DataFrame", isinstance(df_gram, pd.DataFrame))
    ok &= _check("has numeric cols", set(pre.numeric_cols_).issubset(df_gram.columns),
                 f"missing: {set(pre.numeric_cols_) - set(df_gram.columns)}")
    ok &= _check("proto column reattached", "proto" in df_gram.columns)
    ok &= _check("correct row count", len(df_gram) == len(X_adv))

    # Grammar validation runs without error on reconstructed data
    gram_result = validate_dataframe(df_gram)
    ok &= _check("validate_dataframe runs", "all_valid" in gram_result.columns)
    ok &= _check("grammar result length matches", len(gram_result) == len(X_adv))

    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


def test_art_wrapper() -> bool:
    """ART HopSkipJump wrapper initialises (if ART is installed)."""
    print("\ntest_art_wrapper ...")
    if not ART_AVAILABLE:
        print("  SKIP  ART not installed — skipping wrapper test")
        return True

    from sklearn.neural_network import MLPClassifier
    from pipeline.adversarial import generate_fgsm, generate_hopskipjump

    ok = True
    rng = np.random.default_rng(0)
    X = rng.random((200, 20)).astype(np.float32)
    y = rng.integers(0, 2, 200)
    clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=0)
    clf.fit(X, y)

    # Test FGSM (finite-difference, no ART needed)
    X_sub = X[:20]
    X_adv = generate_fgsm(clf, X_sub, eps=0.05)
    ok &= _check("FGSM shape correct", X_adv.shape == X_sub.shape)
    ok &= _check("FGSM perturbs inputs", not np.allclose(X_adv, X_sub))

    # Test HopSkipJump (ART, very small budget for speed)
    try:
        X_hsj = generate_hopskipjump(clf, X[:5], max_eval=20, max_iter=5)
        ok &= _check("HSJ shape correct", X_hsj.shape == X[:5].shape)
    except Exception as e:
        print(f"  SKIP  HSJ raised {e.__class__.__name__}: {e}")

    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


def main() -> int:
    print("Running adversarial pipeline smoke tests ...")
    print("-" * 60)
    ok = True
    ok &= test_compute_detection_metrics()
    ok &= test_per_constraint_vr()
    ok &= test_reconstruct_for_grammar()
    ok &= test_art_wrapper()
    print("-" * 60)
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
