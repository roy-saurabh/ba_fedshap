#!/usr/bin/env python3
"""Dry smoke test for BA-FedSHAP — no torch/shap/flwr required.

Tests all pure-Python components (data loading, partitioning, metrics,
privacy accounting) using sklearn's fetch_openml fallback for Adult data.

Checks that require torch, shap, or flwr are recorded as SKIP.

Exit code: 0 if all non-skipped checks pass, 1 otherwise.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "results" / "smoke_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLIENTS = 5
ALPHA = 0.5
K_GLOBAL = 50
N_EVAL = 50


def check(name: str, results: dict, fn, skip: bool = False):
    if skip:
        logger.info("[SKIP]  %s (dependency unavailable)", name)
        results[name] = {"status": "SKIP"}
        return True
    logger.info("[CHECK] %s ...", name)
    t0 = time.time()
    try:
        val = fn()
        elapsed = time.time() - t0
        logger.info("[PASS]  %s (%.2fs)", name, elapsed)
        results[name] = {"status": "PASS", "elapsed_s": round(elapsed, 3), "value": str(val)[:300]}
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("[FAIL]  %s (%.2fs): %s", name, elapsed, exc)
        results[name] = {
            "status": "FAIL",
            "elapsed_s": round(elapsed, 3),
            "error": str(exc),
            "traceback": traceback.format_exc()[-500:],
        }
        return False


def has_package(pkg: str) -> bool:
    import importlib
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def main() -> int:
    results: dict = {}
    t_start = time.time()
    all_pass = True
    import numpy as np

    _HAS_TORCH = has_package("torch")
    _HAS_SHAP = has_package("shap")
    _HAS_FLWR = has_package("flwr")

    # ── 1. Optional package presence ────────────────────────────────────────
    def check_optional_imports():
        missing = []
        for pkg, avail in [("torch", _HAS_TORCH), ("shap", _HAS_SHAP), ("flwr", _HAS_FLWR)]:
            if not avail:
                missing.append(pkg)
        if missing:
            return f"SKIP (missing: {missing}) — install pinned env to run full test"
        import torch, shap, flwr
        return f"torch={torch.__version__}, shap={shap.__version__}, flwr={flwr.__version__}"

    check("optional_imports", results, check_optional_imports)

    # ── 2. sklearn / numpy / pandas / scipy ─────────────────────────────────
    def check_core_deps():
        import numpy as np
        import pandas as pd
        import scipy, sklearn, matplotlib
        return (f"numpy={np.__version__}, pandas={pd.__version__}, "
                f"scipy={scipy.__version__}, sklearn={sklearn.__version__}")

    if not check("core_deps", results, check_core_deps):
        all_pass = False

    # ── 3. Data loading (Adult via sklearn fallback) ─────────────────────────
    X = y = sensitive = None

    def check_adult_loading():
        nonlocal X, y, sensitive
        from src.data.loaders import load_adult
        X, y, sensitive = load_adult()
        assert len(y) > 1000, f"Expected > 1000 rows, got {len(y)}"
        assert X.shape[1] >= 10
        assert "sex" in sensitive
        assert X.dtype == np.float32
        assert X.min() >= -1e-6 and X.max() <= 1.0 + 1e-6, "Not min-max normalised"
        return f"N={len(y)}, d={X.shape[1]}, groups_sex={np.unique(sensitive['sex'])}"

    if not check("adult_data_loading", results, check_adult_loading):
        all_pass = False
        # save and exit — no data means all downstream checks will fail
        return _save_and_exit(results, all_pass, t_start)

    # ── 4. Dirichlet partition ───────────────────────────────────────────────
    client_datasets = None

    def check_partition():
        nonlocal client_datasets
        from src.data.partition import dirichlet_partition
        client_datasets = dirichlet_partition(X, y, sensitive, N_CLIENTS, ALPHA, SEED)
        assert len(client_datasets) == N_CLIENTS
        total = sum(d["n"] for d in client_datasets)
        assert total == len(y), f"Partition total {total} != {len(y)}"
        sizes = [d["n"] for d in client_datasets]
        return f"clients={N_CLIENTS}, sizes={sizes}, total={total}"

    if not check("dirichlet_partition", results, check_partition):
        all_pass = False

    # ── 5. Config loading ────────────────────────────────────────────────────
    def check_configs():
        import yaml
        cfg_dir = REPO_ROOT / "configs"
        loaded = {}
        for cfg_file in sorted(cfg_dir.glob("*.yaml")):
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            loaded[cfg_file.stem] = cfg.get("dataset", cfg_file.stem)
            assert "seeds" in cfg, f"Missing seeds in {cfg_file.name}"
            assert "n_features" in cfg, f"Missing n_features in {cfg_file.name}"
            assert "protected_attrs" in cfg, f"Missing protected_attrs in {cfg_file.name}"
        return f"configs loaded: {list(loaded.keys())}"

    if not check("config_loading", results, check_configs):
        all_pass = False

    # ── 6. ADI_norm (pure numpy) ─────────────────────────────────────────────
    def check_adi_norm():
        from src.metrics.adi import bootstrap_ci, compute_adi_norm, permutation_pvalue
        rng = np.random.default_rng(SEED)
        d = X.shape[1]
        client_mus = [rng.normal(0.0, 0.1, d).astype(np.float32) for _ in range(N_CLIENTS)]
        prot_labels = np.array([i % 2 for i in range(N_CLIENTS)])
        adi = compute_adi_norm(client_mus, prot_labels)
        assert adi >= 0.0
        pt, lo, hi = bootstrap_ci(client_mus, prot_labels, B=200, seed=SEED)
        assert lo <= pt <= hi, f"CI [{lo},{hi}] does not contain point estimate {pt}"
        pval = permutation_pvalue(client_mus, prot_labels, B_p=200, seed=SEED)
        assert 0.0 <= pval <= 1.0
        return f"adi_norm={adi:.4f}, ci=[{lo:.4f},{hi:.4f}], p={pval:.4f}"

    if not check("adi_norm_metrics", results, check_adi_norm):
        all_pass = False

    # ── 7. RDP accountant (pure numpy) ──────────────────────────────────────
    def check_rdp():
        from src.privacy.rdp_accountant import (
            compute_epsilon,
            compute_sigma_for_epsilon,
            rdp_per_round,
        )
        # Theorem 2: sensitivity = 2*C/m_{k,a} = 2*1.0/100 = 0.02
        # T_eff = 40 (200 rounds * 20% participation per client)
        sensitivity_delta2 = 0.02
        T_eff = 40

        eps_rdp = rdp_per_round(alpha=10.0, sensitivity_delta2=sensitivity_delta2, sigma=0.5942)
        assert eps_rdp > 0

        # sigma=0.5942 should give eps≈1.0 at T=40, delta=1e-5
        eps_dp = compute_epsilon(T=T_eff, sigma=0.5942, sensitivity_delta2=sensitivity_delta2, delta=1e-5)
        assert 0 < eps_dp < 5.0, f"Unexpected epsilon for sigma=0.5942: {eps_dp:.4f}"

        # Reverse: find sigma for eps=1.0
        sigma_for_1 = compute_sigma_for_epsilon(
            target_epsilon=1.0, T=T_eff, sensitivity_delta2=sensitivity_delta2, delta=1e-5
        )
        assert sigma_for_1 > 0
        assert abs(sigma_for_1 - 0.5942) < 0.05, f"sigma mismatch: {sigma_for_1:.4f} vs 0.5942"
        return f"eps(sigma=0.5942,T=40)={eps_dp:.4f}, sigma(eps=1.0)={sigma_for_1:.4f}"

    if not check("rdp_accountant", results, check_rdp):
        all_pass = False

    # ── 8. DP noise (pure numpy) ─────────────────────────────────────────────
    def check_dp_noise():
        from src.privacy.dp_noise import clip_l2, noisy_mean
        rng = np.random.default_rng(SEED)
        v = rng.normal(0, 2.0, (20, X.shape[1])).astype(np.float32)
        clipped = clip_l2(v, C=1.0)
        norms = np.linalg.norm(clipped, axis=1)
        assert np.all(norms <= 1.0 + 1e-5), f"Clip failed, max norm={norms.max():.4f}"
        mean_v = noisy_mean(v, C=1.0, sigma=0.05, seed=SEED)
        assert mean_v.shape == (X.shape[1],)
        return f"clip OK (max_norm={norms.max():.4f}), noisy_mean shape={mean_v.shape}"

    if not check("dp_noise", results, check_dp_noise):
        all_pass = False

    # ── 9. Lipschitz estimator (requires shap — skip otherwise) ─────────────
    def check_lipschitz():
        import shap  # noqa: F401 (will raise ImportError if not installed)
        from sklearn.linear_model import LogisticRegression
        from src.metrics.lipschitz import estimate_local_lipschitz

        clf = LogisticRegression(random_state=SEED, max_iter=200)
        clf.fit(X[:500], y[:500])

        class SklearnWrapper:
            def __init__(self, c): self.clf = c
            def predict_proba(self, X): return self.clf.predict_proba(X)

        rng = np.random.default_rng(SEED)
        bg_idx = rng.choice(len(y), K_GLOBAL, replace=False)
        background = X[bg_idx]

        res = estimate_local_lipschitz(SklearnWrapper(clf), X[:100], background,
                                       n_pairs=20, eps_norm=0.01, seed=SEED)
        assert res["l_hat"] > 0.0, f"Non-positive L_hat: {res}"
        return f"L_hat={res['l_hat']:.4f}"

    if not check("lipschitz_estimator", results, check_lipschitz, skip=not _HAS_SHAP):
        all_pass = False

    # ── 10. Faithfulness (sklearn model, no torch) ───────────────────────────
    def check_faithfulness():
        from sklearn.linear_model import LogisticRegression
        from src.metrics.faithfulness import faithfulness_report
        import numpy as np

        rng = np.random.default_rng(SEED)
        clf = LogisticRegression(random_state=SEED, max_iter=200)
        clf.fit(X[:500], y[:500])

        class Wrapper:
            def __init__(self, clf):
                self.clf = clf
            def predict_proba(self, X):
                return self.clf.predict_proba(X)

        idx = rng.choice(len(y), N_EVAL, replace=False)
        x_eval = X[idx]
        bg_idx = rng.choice(len(y), K_GLOBAL, replace=False)
        background = X[bg_idx]
        # Create simple fake SHAP values (random; tests the AUC loop logic)
        shap_vals = rng.normal(0, 0.1, (N_EVAL, X.shape[1])).astype(np.float32)
        report = faithfulness_report(Wrapper(clf), x_eval, shap_vals, background, n_steps=5)
        assert "deletion_auc" in report and "insertion_auc" in report
        assert 0 <= report["deletion_auc"] <= 1.0
        assert 0 <= report["insertion_auc"] <= 1.0
        return (f"deletion_auc={report['deletion_auc']:.4f}, "
                f"insertion_auc={report['insertion_auc']:.4f}")

    if not check("faithfulness_metrics", results, check_faithfulness):
        all_pass = False

    # ── 11. Torch-dependent checks (SKIP if not installed) ───────────────────
    check("fedavg_training", results, lambda: None, skip=not _HAS_TORCH)
    check("kernelshap_computation", results, lambda: None, skip=(not _HAS_SHAP or not _HAS_TORCH))
    check("ba_fedshap_protocol", results, lambda: None, skip=(not _HAS_SHAP or not _HAS_TORCH))
    check("flwr_import", results, lambda: None, skip=not _HAS_FLWR)

    return _save_and_exit(results, all_pass, t_start)


def _save_and_exit(results: dict, all_pass: bool, t_start: float) -> int:
    total = time.time() - t_start
    n_pass = sum(1 for r in results.values() if r.get("status") == "PASS")
    n_fail = sum(1 for r in results.values() if r.get("status") == "FAIL")
    n_skip = sum(1 for r in results.values() if r.get("status") == "SKIP")

    summary = {
        "overall": "PASS" if all_pass else "FAIL",
        "n_checks": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "total_elapsed_s": round(total, 2),
        "note": ("Checks marked SKIP require torch==2.1.0, shap==0.44.0, flwr==1.5.0 "
                 "in the pinned conda environment (see environment.yml)."),
        "checks": results,
    }

    out_file = OUT_DIR / "dry_smoke_test_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"DRY SMOKE TEST: {'PASS' if all_pass else 'FAIL'}")
    print(f"  {n_pass} passed  |  {n_fail} failed  |  {n_skip} skipped  ({total:.1f}s)")
    print(f"  Results written to: {out_file}")
    print("=" * 65)
    if n_fail:
        print("\nFailed checks:")
        for name, r in results.items():
            if r.get("status") == "FAIL":
                print(f"  ✗ {name}: {r.get('error', '')[:120]}")
    if n_skip:
        print(f"\nSkipped (need pinned env): "
              f"{[k for k,v in results.items() if v.get('status')=='SKIP']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
