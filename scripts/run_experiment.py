#!/usr/bin/env python3
"""Main experiment runner for BA-FedSHAP.

Usage:
    python scripts/run_experiment.py --config configs/adult.yaml
    python scripts/run_experiment.py --config configs/adult.yaml --seed 42 --alpha 0.5

Outputs are saved to:
    results/raw/{dataset}/seed_{seed}/alpha_{alpha}/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root or scripts/ dir
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_dataset
from src.data.partition import dirichlet_partition
from src.explainers.baselines import (
    centralized_oracle_shap,
    gradient_federated,
    kmeans_background_shap,
    naive_aggregated_shap,
    shared_background_shap,
)
from src.explainers.kernelshap_wrapper import subsample_background
from src.federated.ba_fedshap import BAFedSHAP
from src.federated.training import FederatedTrainer, build_model
from src.metrics.adi import bootstrap_ci, compute_adi_norm, compute_l1_to_global, compute_spearman_rank, permutation_pvalue
from src.metrics.faithfulness import faithfulness_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BA-FedSHAP experiment.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    p.add_argument("--seed", type=int, default=None, help="Override seed (runs single seed).")
    p.add_argument("--alpha", type=float, default=None, help="Override Dirichlet alpha.")
    p.add_argument("--epsilon", type=str, default=None,
                   help="Override DP epsilon (e.g. '1', '4', 'inf').")
    p.add_argument("--no-baselines", action="store_true", help="Skip baseline methods.")
    p.add_argument("--results-dir", type=str, default="results/raw",
                   help="Root directory for raw results.")
    # CPU-friendly overrides
    p.add_argument("--n-clients", type=int, default=None, help="Override n_clients.")
    p.add_argument("--fl-rounds", type=int, default=None, help="Override fl_rounds.")
    p.add_argument("--coalitions", type=int, default=None, help="Override kernelshap_coalitions.")
    p.add_argument("--background-size", type=int, default=None, help="Override background_size.")
    p.add_argument("--max-client-eval", type=int, default=None,
                   help="Max samples per client group for SHAP (subsamples large clients).")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_single(cfg: dict, seed: int, alpha: float, results_dir: Path, skip_baselines: bool,
               n_clients_override: int = None, fl_rounds_override: int = None,
               coalitions_override: int = None, background_override: int = None,
               max_client_eval: int = None) -> None:
    dataset_name = cfg["dataset"]
    n_clients = n_clients_override if n_clients_override else cfg.get("n_clients", 50)
    fl_rounds = fl_rounds_override if fl_rounds_override else cfg.get("fl_rounds", 200)
    local_epochs = cfg.get("local_epochs", 5)
    lr = cfg.get("lr", 0.01)
    momentum = cfg.get("momentum", 0.9)
    K_global = background_override if background_override else cfg.get("background_size", 200)
    M = coalitions_override if coalitions_override else cfg.get("kernelshap_coalitions", 2048)
    dp_sigmas = cfg.get("dp_sigmas", {"inf": 0.0})
    model_name = cfg.get("model", "logreg")
    C = cfg.get("dp_clip_c", 1.0)
    z_tau = cfg.get("drift_z_tau", 3.5)
    protected_attrs = cfg.get("protected_attrs", [])

    # Output path
    out_dir = results_dir / dataset_name / f"seed_{seed}" / f"alpha_{alpha:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Dataset=%s | seed=%d | alpha=%.2f ===", dataset_name, seed, alpha)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.time()
    kwargs = {}
    if dataset_name in ("acs_income", "acs_pubcov"):
        kwargs["survey_year"] = cfg.get("acs_survey_year", "2018")
        kwargs["states"] = cfg.get("acs_states", None)

    X, y, sensitive = load_dataset(dataset_name, **kwargs)
    logger.info("Loaded %s: N=%d, d=%d, pos_rate=%.3f", dataset_name, len(y), X.shape[1], y.mean())

    # ------------------------------------------------------------------
    # 2. Partition into clients
    # ------------------------------------------------------------------
    # Use first protected attribute for partitioning group labels
    prot_key = list(sensitive.keys())[0] if sensitive else None
    prot_arr = sensitive[prot_key] if prot_key else np.zeros(len(y), dtype=int)

    client_datasets = dirichlet_partition(
        X=X, y=y, sensitive=sensitive,
        n_clients=n_clients, alpha=alpha, seed=seed,
    )

    # ------------------------------------------------------------------
    # 3. Train global model via FedAvg
    # ------------------------------------------------------------------
    n_features = X.shape[1]
    global_model = build_model(model_name, n_features)

    trainer = FederatedTrainer(seed=seed)
    global_model = trainer.fit(
        client_datasets=client_datasets,
        global_model=global_model,
        rounds=fl_rounds,
        local_epochs=local_epochs,
        lr=lr,
        momentum=momentum,
    )
    eval_metrics = trainer.evaluate(global_model, client_datasets)
    logger.info("Global model: acc=%.4f", eval_metrics["accuracy_mean"])

    # ------------------------------------------------------------------
    # 4. Reference data for backgrounds (10% holdout)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    n_ref = max(K_global, int(0.1 * len(y)))
    ref_idx = rng.choice(len(y), size=n_ref, replace=False)
    ref_data = X[ref_idx]
    ref_prot = prot_arr[ref_idx]

    # Evaluation set (small subset for speed)
    n_eval = min(500, len(y) - n_ref)
    remaining = np.setdiff1d(np.arange(len(y)), ref_idx)
    eval_idx = rng.choice(remaining, size=n_eval, replace=False)
    x_eval = X[eval_idx]

    background_global = subsample_background(ref_data, K=K_global, seed=seed)

    # ------------------------------------------------------------------
    # 5. Run BA-FedSHAP at each DP epsilon
    # ------------------------------------------------------------------
    results = {
        "dataset": dataset_name,
        "seed": seed,
        "alpha": alpha,
        "n_features": n_features,
        "n_clients": n_clients,
        "fl_rounds": fl_rounds,
        "eval_accuracy": eval_metrics["accuracy_mean"],
        "ba_fedshap": {},
        "baselines": {},
    }

    # Subsample large client datasets for SHAP to keep CPU runtimes tractable
    client_datasets_eval = client_datasets
    if max_client_eval is not None:
        rng_sub = np.random.default_rng(seed + 9999)
        client_datasets_eval = []
        for ds in client_datasets:
            if ds["n"] > max_client_eval:
                idx = rng_sub.choice(ds["n"], size=max_client_eval, replace=False)
                sub = {
                    "X": ds["X"][idx],
                    "y": ds["y"][idx],
                    "sensitive": {k: v[idx] for k, v in ds["sensitive"].items()},
                    "n": max_client_eval,
                    "client_id": ds.get("client_id"),
                }
                client_datasets_eval.append(sub)
            else:
                client_datasets_eval.append(ds)

    protocol = BAFedSHAP(K_global=K_global, n_coalitions=M, seed=seed)

    for eps_label, sigma in dp_sigmas.items():
        logger.info("-- BA-FedSHAP epsilon=%s (sigma=%.4f) --", eps_label, sigma)
        run_cfg = {
            "background_size": K_global,
            "dp_clip_c": C,
            "dp_sigma": float(sigma),
            "drift_z_tau": z_tau,
            "kernelshap_coalitions": M,
            "seed": seed,
        }

        t_start = time.time()
        try:
            proto_result = protocol.run_full_protocol(
                client_datasets=client_datasets_eval,
                global_model=global_model,
                ref_data=ref_data,
                ref_protected=ref_prot,
                config=run_cfg,
            )
            runtime_s = time.time() - t_start

            # ADI metrics
            client_mus = [
                np.array(list(cs["group_mus"].values())).mean(axis=0)
                for cs in proto_result["client_summaries"]
                if cs["group_mus"]
            ]
            prot_labels = prot_arr[eval_idx[:len(client_mus)]] if len(client_mus) > 0 else np.array([])

            adi_pt = proto_result["adi_norm"]
            results["ba_fedshap"][str(eps_label)] = {
                "adi_norm": adi_pt,
                "n_clients_used": proto_result["n_clients_used"],
                "outlier_clients": proto_result["outlier_clients"],
                "runtime_s": runtime_s,
            }
            logger.info(
                "epsilon=%s: ADI_norm=%.4f, clients_used=%d, time=%.1fs",
                eps_label, adi_pt, proto_result["n_clients_used"], runtime_s,
            )
        except Exception as exc:
            logger.error("BA-FedSHAP epsilon=%s failed: %s", eps_label, exc)
            results["ba_fedshap"][str(eps_label)] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 6. Baselines
    # ------------------------------------------------------------------
    if not skip_baselines:
        logger.info("-- Running baselines --")

        # Centralized oracle
        try:
            t_start = time.time()
            shap_oracle = centralized_oracle_shap(
                pooled_data=X, model=global_model, x_eval=x_eval,
                background_size=K_global, n_coalitions=M, seed=seed,
            )
            results["baselines"]["centralized_oracle"] = {
                "shap_mean_abs_norm": float(np.abs(shap_oracle).mean()),
                "runtime_s": time.time() - t_start,
            }
        except Exception as exc:
            logger.error("Centralized oracle failed: %s", exc)
            results["baselines"]["centralized_oracle"] = {"error": str(exc)}

        # Naive aggregated
        try:
            t_start = time.time()
            shap_naive = naive_aggregated_shap(
                client_datasets=client_datasets, model=global_model, x_eval=x_eval,
                background_size=K_global, n_coalitions=M, seed=seed,
            )
            results["baselines"]["naive_aggregated"] = {
                "shap_mean_abs_norm": float(np.abs(shap_naive).mean()),
                "runtime_s": time.time() - t_start,
            }
        except Exception as exc:
            logger.error("Naive aggregated failed: %s", exc)
            results["baselines"]["naive_aggregated"] = {"error": str(exc)}

        # Shared background
        try:
            t_start = time.time()
            shap_shared = shared_background_shap(
                client_datasets=client_datasets, model=global_model, x_eval=x_eval,
                background_pool=background_global, n_coalitions=M, seed=seed,
            )
            results["baselines"]["shared_background"] = {
                "shap_mean_abs_norm": float(np.abs(shap_shared).mean()),
                "runtime_s": time.time() - t_start,
            }
        except Exception as exc:
            logger.error("Shared background failed: %s", exc)
            results["baselines"]["shared_background"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    results["total_runtime_s"] = time.time() - t0
    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_file)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42])
    alphas = [args.alpha] if args.alpha is not None else cfg.get("dirichlet_alphas", [0.5])
    results_dir = Path(args.results_dir)

    for seed in seeds:
        for alpha in alphas:
            try:
                run_single(
                    cfg, int(seed), float(alpha), results_dir, args.no_baselines,
                    n_clients_override=args.n_clients,
                    fl_rounds_override=args.fl_rounds,
                    coalitions_override=args.coalitions,
                    background_override=args.background_size,
                    max_client_eval=args.max_client_eval,
                )
            except Exception as exc:
                logger.error("FAILED seed=%d alpha=%.2f: %s", seed, alpha, exc, exc_info=True)


if __name__ == "__main__":
    main()
