#!/usr/bin/env python3
"""Fast smoke test for BA-FedSHAP.

Uses the Adult dataset with minimal settings to verify all components
run without error in < 5 minutes.

Settings:
    n_clients=5, rounds=3, seeds=[42], alpha=0.5, n_eval=50, K=50, M=64

Output: results/smoke_test/smoke_test_results.json
Exit code: 0 on PASS, 1 on FAIL
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SMOKE_DIR = REPO_ROOT / "results" / "smoke_test"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

# Smoke test parameters (fast)
N_CLIENTS = 5
ROUNDS = 3
SEED = 42
ALPHA = 0.5
N_EVAL = 50
K_GLOBAL = 50
M_COALITIONS = 64
LOCAL_EPOCHS = 2


def check(name: str, results: dict, fn):
    """Run a named check and record pass/fail."""
    logger.info("[CHECK] %s ...", name)
    t0 = time.time()
    try:
        val = fn()
        elapsed = time.time() - t0
        logger.info("[PASS]  %s (%.2fs)", name, elapsed)
        results[name] = {"status": "PASS", "elapsed_s": elapsed, "value": str(val)[:200]}
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        logger.error("[FAIL]  %s (%.2fs): %s", name, elapsed, exc)
        results[name] = {"status": "FAIL", "elapsed_s": elapsed, "error": str(exc), "traceback": tb}
        return False


def main() -> int:
    results = {}
    t_start = time.time()
    all_pass = True

    # ------------------------------------------------------------------
    # 1. Imports
    # ------------------------------------------------------------------
    import numpy as np

    def check_imports():
        import shap
        import torch
        import flwr
        return f"shap={shap.__version__}, torch={torch.__version__}"

    if not check("imports", results, check_imports):
        all_pass = False

    # ------------------------------------------------------------------
    # 2. Data loading
    # ------------------------------------------------------------------
    X = y = sensitive = None

    def check_data_loading():
        nonlocal X, y, sensitive
        from src.data.loaders import load_adult
        X, y, sensitive = load_adult()
        assert len(y) > 1000, f"Expected > 1000 rows, got {len(y)}"
        assert X.shape[1] > 0
        assert "sex" in sensitive
        return f"N={len(y)}, d={X.shape[1]}"

    if not check("data_loading", results, check_data_loading):
        all_pass = False
        # Cannot proceed without data
        _save_and_exit(results, all_pass, t_start)

    # ------------------------------------------------------------------
    # 3. Data partitioning
    # ------------------------------------------------------------------
    client_datasets = None

    def check_partition():
        nonlocal client_datasets
        from src.data.partition import dirichlet_partition
        client_datasets = dirichlet_partition(X, y, sensitive, N_CLIENTS, ALPHA, SEED)
        assert len(client_datasets) == N_CLIENTS
        assert all(d["n"] >= 0 for d in client_datasets)
        sizes = [d["n"] for d in client_datasets]
        return f"clients={len(client_datasets)}, sizes={sizes}"

    if not check("dirichlet_partition", results, check_partition):
        all_pass = False

    # ------------------------------------------------------------------
    # 4. Model build
    # ------------------------------------------------------------------
    model = None

    def check_model():
        nonlocal model
        from src.federated.training import build_model
        model = build_model("logreg", X.shape[1])
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        return f"model built, proba shape={proba.shape}"

    if not check("model_build", results, check_model):
        all_pass = False

    # ------------------------------------------------------------------
    # 5. FedAvg training
    # ------------------------------------------------------------------
    def check_fedavg():
        nonlocal model
        from src.federated.training import FederatedTrainer
        trainer = FederatedTrainer(seed=SEED)
        model = trainer.fit(
            client_datasets=client_datasets,
            global_model=model,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
        )
        acc = trainer.evaluate(model, client_datasets)["accuracy_mean"]
        assert 0 < acc <= 1.0, f"Unexpected accuracy: {acc}"
        return f"accuracy={acc:.4f}"

    if not check("fedavg_training", results, check_fedavg):
        all_pass = False

    # ------------------------------------------------------------------
    # 6. KernelSHAP computation
    # ------------------------------------------------------------------
    shap_values = None
    background = None

    def check_kernelshap():
        nonlocal shap_values, background
        from src.explainers.kernelshap_wrapper import compute_shap, subsample_background
        rng = np.random.default_rng(SEED)
        ref_idx = rng.choice(len(y), size=K_GLOBAL, replace=False)
        background = X[ref_idx].astype(np.float32)
        eval_idx = rng.choice(len(y), size=N_EVAL, replace=False)
        x_eval = X[eval_idx]
        shap_values = compute_shap(model, x_eval, background, n_coalitions=M_COALITIONS, seed=SEED)
        assert shap_values.shape == (N_EVAL, X.shape[1]), f"Unexpected shape: {shap_values.shape}"
        return f"shap shape={shap_values.shape}, mean_abs={np.abs(shap_values).mean():.4f}"

    if not check("kernelshap", results, check_kernelshap):
        all_pass = False

    # ------------------------------------------------------------------
    # 7. BA-FedSHAP protocol
    # ------------------------------------------------------------------
    def check_ba_fedshap():
        from src.federated.ba_fedshap import BAFedSHAP
        rng = np.random.default_rng(SEED)
        ref_idx = rng.choice(len(y), size=K_GLOBAL, replace=False)
        ref_data = X[ref_idx].astype(np.float32)
        ref_prot = sensitive["sex"][ref_idx]

        protocol = BAFedSHAP(K_global=K_GLOBAL, n_coalitions=M_COALITIONS, seed=SEED)
        run_cfg = {
            "background_size": K_GLOBAL,
            "dp_clip_c": 1.0,
            "dp_sigma": 0.0,
            "drift_z_tau": 3.5,
            "kernelshap_coalitions": M_COALITIONS,
            "seed": SEED,
        }
        proto_result = protocol.run_full_protocol(
            client_datasets=client_datasets,
            global_model=model,
            ref_data=ref_data,
            ref_protected=ref_prot,
            config=run_cfg,
        )
        assert "adi_norm" in proto_result
        assert "global_mu" in proto_result
        assert proto_result["n_clients_used"] > 0
        return f"adi_norm={proto_result['adi_norm']:.4f}, clients_used={proto_result['n_clients_used']}"

    if not check("ba_fedshap_protocol", results, check_ba_fedshap):
        all_pass = False

    # ------------------------------------------------------------------
    # 8. Baselines
    # ------------------------------------------------------------------
    def check_baselines():
        from src.explainers.baselines import (
            centralized_oracle_shap,
            naive_aggregated_shap,
            shared_background_shap,
        )
        rng = np.random.default_rng(SEED)
        eval_idx = rng.choice(len(y), size=N_EVAL, replace=False)
        x_eval = X[eval_idx]

        shap_oracle = centralized_oracle_shap(
            X, model, x_eval, background_size=K_GLOBAL, n_coalitions=M_COALITIONS, seed=SEED
        )
        shap_naive = naive_aggregated_shap(
            client_datasets, model, x_eval, background_size=K_GLOBAL,
            n_coalitions=M_COALITIONS, seed=SEED,
        )
        shap_shared = shared_background_shap(
            client_datasets, model, x_eval, background_pool=background,
            n_coalitions=M_COALITIONS, seed=SEED,
        )
        assert shap_oracle.shape[1] == X.shape[1]
        assert shap_naive.shape[1] == X.shape[1]
        assert shap_shared.shape[1] == X.shape[1]
        return "oracle+naive+shared OK"

    if not check("baselines", results, check_baselines):
        all_pass = False

    # ------------------------------------------------------------------
    # 9. ADI metrics
    # ------------------------------------------------------------------
    def check_adi_metrics():
        from src.metrics.adi import bootstrap_ci, compute_adi_norm, permutation_pvalue
        # Mock client mus
        rng = np.random.default_rng(SEED)
        client_mus = [rng.normal(0, 0.1, X.shape[1]) for _ in range(N_CLIENTS)]
        prot_labels = np.array([i % 2 for i in range(N_CLIENTS)])
        adi = compute_adi_norm(client_mus, prot_labels)
        pt, lo, hi = bootstrap_ci(client_mus, prot_labels, B=100, seed=SEED)
        pval = permutation_pvalue(client_mus, prot_labels, B_p=100, seed=SEED)
        assert 0 <= pval <= 1.0
        return f"adi={adi:.4f}, ci=[{lo:.4f},{hi:.4f}], pval={pval:.4f}"

    if not check("adi_metrics", results, check_adi_metrics):
        all_pass = False

    # ------------------------------------------------------------------
    # 10. Faithfulness metrics
    # ------------------------------------------------------------------
    def check_faithfulness():
        from src.metrics.faithfulness import faithfulness_report
        if shap_values is None or background is None:
            raise RuntimeError("SHAP values not computed (kernelshap check failed).")
        rng = np.random.default_rng(SEED)
        eval_idx = rng.choice(len(y), size=N_EVAL, replace=False)
        x_eval = X[eval_idx]
        report = faithfulness_report(model, x_eval, shap_values, background, n_steps=5)
        assert "deletion_auc" in report
        assert "insertion_auc" in report
        return f"del_auc={report['deletion_auc']:.4f}, ins_auc={report['insertion_auc']:.4f}"

    if not check("faithfulness", results, check_faithfulness):
        all_pass = False

    # ------------------------------------------------------------------
    # 11. RDP accountant
    # ------------------------------------------------------------------
    def check_rdp():
        from src.privacy.rdp_accountant import compute_epsilon, compute_sigma_for_epsilon
        eps = compute_epsilon(T=ROUNDS, sigma=0.141, sensitivity_delta2=0.02, delta=1e-5)
        sigma = compute_sigma_for_epsilon(target_epsilon=1.0, T=ROUNDS,
                                         sensitivity_delta2=0.02, delta=1e-5)
        assert eps > 0
        assert sigma > 0
        return f"eps={eps:.4f}, sigma_for_eps1={sigma:.4f}"

    if not check("rdp_accountant", results, check_rdp):
        all_pass = False

    # ------------------------------------------------------------------
    # 12. DP noise
    # ------------------------------------------------------------------
    def check_dp_noise():
        from src.privacy.dp_noise import clip_l2, noisy_mean
        rng = np.random.default_rng(SEED)
        v = rng.normal(0, 1, (10, X.shape[1])).astype(np.float32)
        clipped = clip_l2(v, C=1.0)
        assert np.all(np.linalg.norm(clipped, axis=1) <= 1.0 + 1e-5)
        mean = noisy_mean(v, C=1.0, sigma=0.1, seed=SEED)
        assert mean.shape == (X.shape[1],)
        return f"clip OK, noisy_mean shape={mean.shape}"

    if not check("dp_noise", results, check_dp_noise):
        all_pass = False

    return _save_and_exit(results, all_pass, t_start)


def _save_and_exit(results: dict, all_pass: bool, t_start: float) -> int:
    total_time = time.time() - t_start
    n_pass = sum(1 for r in results.values() if r.get("status") == "PASS")
    n_fail = sum(1 for r in results.values() if r.get("status") == "FAIL")

    summary = {
        "overall": "PASS" if all_pass else "FAIL",
        "n_checks": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "total_elapsed_s": total_time,
        "checks": results,
    }

    out_file = SMOKE_DIR / "smoke_test_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SMOKE TEST: {'PASS' if all_pass else 'FAIL'}")
    print(f"Checks: {n_pass}/{n_fail + n_pass} passed ({total_time:.1f}s)")
    print(f"Results: {out_file}")
    print("=" * 60)

    if n_fail > 0:
        print("\nFailed checks:")
        for name, r in results.items():
            if r.get("status") == "FAIL":
                print(f"  - {name}: {r.get('error', '')[:100]}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
