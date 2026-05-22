#!/usr/bin/env python3
"""Run stability experiments: cross-client and cross-seed Spearman correlations.

Measures how consistent the SHAP feature rankings are:
  (a) Across clients (within a seed)
  (b) Across seeds (same client configuration)

Results saved to results/raw/{dataset}/stability/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_dataset
from src.data.partition import dirichlet_partition
from src.explainers.kernelshap_wrapper import compute_shap, subsample_background
from src.federated.training import FederatedTrainer, build_model
from src.metrics.adi import compute_spearman_rank

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BA-FedSHAP stability experiment.")
    p.add_argument("--config", type=str, default="configs/adult.yaml")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--results-dir", type=str, default="results/raw")
    return p.parse_args()


def run_stability(cfg: dict, alpha: float, results_dir: Path) -> None:
    dataset_name = cfg["dataset"]
    seeds = cfg.get("seeds", [42, 123, 456, 789, 1024])
    n_clients = cfg.get("n_clients", 50)
    fl_rounds = cfg.get("fl_rounds", 200)
    local_epochs = cfg.get("local_epochs", 5)
    K_global = cfg.get("background_size", 200)
    M = cfg.get("kernelshap_coalitions", 2048)
    model_name = cfg.get("model", "logreg")

    out_dir = results_dir / dataset_name / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    kwargs = {}
    if dataset_name in ("acs_income", "acs_pubcov"):
        kwargs["survey_year"] = cfg.get("acs_survey_year", "2018")
        kwargs["states"] = cfg.get("acs_states", None)
    X, y, sensitive = load_dataset(dataset_name, **kwargs)
    n_features = X.shape[1]

    global_shap_by_seed = []
    per_client_shap_by_seed = []

    for seed in seeds:
        logger.info("Stability: seed=%d, alpha=%.2f", seed, alpha)
        rng = np.random.default_rng(seed)

        client_datasets = dirichlet_partition(X, y, sensitive, n_clients, alpha, seed)
        model = build_model(model_name, n_features)
        trainer = FederatedTrainer(seed=seed)
        model = trainer.fit(client_datasets, model, rounds=fl_rounds,
                            local_epochs=local_epochs)

        # Reference background
        ref_idx = rng.choice(len(y), size=K_global, replace=False)
        background = X[ref_idx].astype(np.float32)

        # Eval set
        eval_idx = rng.choice(len(y), size=min(200, len(y)), replace=False)
        x_eval = X[eval_idx]

        # Global SHAP
        global_shap = compute_shap(model, x_eval, background, n_coalitions=M, seed=seed)
        global_mu = global_shap.mean(axis=0)
        global_shap_by_seed.append(global_mu)

        # Per-client SHAP mean vectors
        client_mus = []
        for k, ds in enumerate(client_datasets[:10]):  # sample 10 clients for speed
            if ds["n"] < 5:
                continue
            bg_k = subsample_background(ds["X"], K=min(50, ds["n"]), seed=seed + k)
            shap_k = compute_shap(model, x_eval, bg_k, n_coalitions=min(M, 512), seed=seed + k)
            client_mus.append(shap_k.mean(axis=0))

        per_client_shap_by_seed.append(client_mus)

    # Cross-seed Spearman (pairwise)
    n_seeds = len(global_shap_by_seed)
    cross_seed_rhos = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            rho, _ = stats.spearmanr(
                np.abs(global_shap_by_seed[i]),
                np.abs(global_shap_by_seed[j]),
            )
            cross_seed_rhos.append(float(rho))

    # Cross-client Spearman (within seed 0)
    cross_client_rhos = []
    if per_client_shap_by_seed and len(per_client_shap_by_seed[0]) >= 2:
        mus = per_client_shap_by_seed[0]
        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                rho, _ = stats.spearmanr(np.abs(mus[i]), np.abs(mus[j]))
                cross_client_rhos.append(float(rho))

    results = {
        "dataset": dataset_name,
        "alpha": alpha,
        "seeds": list(seeds),
        "cross_seed_spearman_mean": float(np.mean(cross_seed_rhos)) if cross_seed_rhos else float("nan"),
        "cross_seed_spearman_std": float(np.std(cross_seed_rhos)) if cross_seed_rhos else float("nan"),
        "cross_client_spearman_mean": float(np.mean(cross_client_rhos)) if cross_client_rhos else float("nan"),
        "cross_client_spearman_std": float(np.std(cross_client_rhos)) if cross_client_rhos else float("nan"),
        "n_cross_seed_pairs": len(cross_seed_rhos),
        "n_cross_client_pairs": len(cross_client_rhos),
    }

    out_file = out_dir / f"stability_alpha{alpha:.2f}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Stability results: cross-seed rho=%.4f, cross-client rho=%.4f",
                results["cross_seed_spearman_mean"],
                results["cross_client_spearman_mean"])
    logger.info("Saved: %s", out_file)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_stability(cfg, args.alpha, Path(args.results_dir))


if __name__ == "__main__":
    main()
