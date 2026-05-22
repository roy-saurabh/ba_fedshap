#!/usr/bin/env python3
"""Run deletion/insertion AUC faithfulness evaluation for all baselines.

Results saved to results/raw/{dataset}/faithfulness/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_dataset
from src.data.partition import dirichlet_partition
from src.explainers.baselines import (
    centralized_oracle_shap,
    kmeans_background_shap,
    naive_aggregated_shap,
    shared_background_shap,
)
from src.explainers.kernelshap_wrapper import compute_shap, subsample_background
from src.federated.ba_fedshap import BAFedSHAP
from src.federated.training import FederatedTrainer, build_model
from src.metrics.faithfulness import faithfulness_report

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BA-FedSHAP faithfulness experiment.")
    p.add_argument("--config", type=str, default="configs/adult.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--results-dir", type=str, default="results/raw")
    return p.parse_args()


def run_faithfulness(cfg: dict, seed: int, alpha: float, results_dir: Path) -> None:
    dataset_name = cfg["dataset"]
    n_clients = cfg.get("n_clients", 50)
    fl_rounds = cfg.get("fl_rounds", 200)
    local_epochs = cfg.get("local_epochs", 5)
    K_global = cfg.get("background_size", 200)
    M = cfg.get("kernelshap_coalitions", 2048)
    model_name = cfg.get("model", "logreg")

    out_dir = results_dir / dataset_name / f"seed_{seed}" / "faithfulness"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if dataset_name in ("acs_income", "acs_pubcov"):
        kwargs["survey_year"] = cfg.get("acs_survey_year", "2018")
        kwargs["states"] = cfg.get("acs_states", None)

    X, y, sensitive = load_dataset(dataset_name, **kwargs)
    n_features = X.shape[1]
    rng = np.random.default_rng(seed)

    client_datasets = dirichlet_partition(X, y, sensitive, n_clients, alpha, seed)
    model = build_model(model_name, n_features)
    trainer = FederatedTrainer(seed=seed)
    model = trainer.fit(client_datasets, model, rounds=fl_rounds, local_epochs=local_epochs)

    ref_idx = rng.choice(len(y), size=K_global, replace=False)
    background = X[ref_idx].astype(np.float32)
    eval_idx = rng.choice(len(y), size=min(300, len(y)), replace=False)
    x_eval = X[eval_idx]

    results = {"dataset": dataset_name, "seed": seed, "alpha": alpha, "methods": {}}

    # BA-FedSHAP (no DP)
    protocol = BAFedSHAP(K_global=K_global, n_coalitions=M, seed=seed)
    prot_key = list(sensitive.keys())[0]
    ref_prot = sensitive[prot_key][ref_idx]
    backgrounds = protocol.construct_backgrounds(background, ref_prot, K_global, seed)
    q_pool = backgrounds["__global__"]

    # Compute BA-FedSHAP SHAP values
    baf_shap = compute_shap(model, x_eval, q_pool, n_coalitions=M, seed=seed)
    faith_baf = faithfulness_report(model, x_eval, baf_shap, background)
    results["methods"]["ba_fedshap"] = faith_baf

    # Centralized oracle
    oracle_shap = centralized_oracle_shap(X, model, x_eval, K_global, M, seed)
    faith_oracle = faithfulness_report(model, x_eval, oracle_shap, background)
    results["methods"]["centralized_oracle"] = faith_oracle

    # Naive aggregated
    naive_shap = naive_aggregated_shap(client_datasets, model, x_eval,
                                       background_size=K_global, n_coalitions=M, seed=seed)
    faith_naive = faithfulness_report(model, x_eval, naive_shap, background)
    results["methods"]["naive_aggregated"] = faith_naive

    # Shared background
    shared_shap = shared_background_shap(client_datasets, model, x_eval,
                                          background_pool=background, n_coalitions=M, seed=seed)
    faith_shared = faithfulness_report(model, x_eval, shared_shap, background)
    results["methods"]["shared_background"] = faith_shared

    # k-means background
    kmeans_shap = kmeans_background_shap(client_datasets, model, x_eval,
                                          k=K_global, n_coalitions=M, seed=seed)
    faith_kmeans = faithfulness_report(model, x_eval, kmeans_shap, background)
    results["methods"]["kmeans_background"] = faith_kmeans

    out_file = out_dir / f"faithfulness_alpha{alpha:.2f}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    for method, metrics in results["methods"].items():
        logger.info("%s: deletion_auc=%.4f, insertion_auc=%.4f",
                    method, metrics["deletion_auc"], metrics["insertion_auc"])

    logger.info("Saved: %s", out_file)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_faithfulness(cfg, args.seed, args.alpha, Path(args.results_dir))


if __name__ == "__main__":
    main()
