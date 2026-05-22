#!/usr/bin/env python3
"""Run parameter and label randomization sanity checks.

Results saved to results/raw/{dataset}/sanity/
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
from src.explainers.kernelshap_wrapper import compute_shap, subsample_background
from src.federated.training import FederatedTrainer, LogisticRegressionModel, MLPModel, build_model
from src.metrics.sanity import label_randomization_check, parameter_randomization_check

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BA-FedSHAP sanity checks.")
    p.add_argument("--config", type=str, default="configs/adult.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--results-dir", type=str, default="results/raw")
    return p.parse_args()


def run_sanity(cfg: dict, seed: int, alpha: float, results_dir: Path) -> None:
    dataset_name = cfg["dataset"]
    n_clients = cfg.get("n_clients", 50)
    fl_rounds = cfg.get("fl_rounds", 200)
    local_epochs = cfg.get("local_epochs", 5)
    K_global = cfg.get("background_size", 200)
    M = cfg.get("kernelshap_coalitions", 2048)
    model_name = cfg.get("model", "logreg")

    out_dir = results_dir / dataset_name / f"seed_{seed}" / "sanity"
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
    eval_idx = rng.choice(len(y), size=min(200, len(y)), replace=False)
    x_eval = X[eval_idx]

    # Compute original SHAP values
    shap_orig = compute_shap(model, x_eval, background, n_coalitions=min(M, 512), seed=seed)

    results = {
        "dataset": dataset_name,
        "seed": seed,
        "alpha": alpha,
    }

    # 1. Parameter randomization check
    logger.info("Running parameter randomization check...")
    param_rand = parameter_randomization_check(
        model=model,
        x_eval=x_eval,
        background=background,
        shap_values_original=shap_orig,
        n_coalitions=256,
        seed=seed,
    )
    results["parameter_randomization"] = param_rand
    logger.info("Param rand: cosine_sim=%.4f, pass=%s",
                param_rand["cosine_similarity"], param_rand["pass"])

    # 2. Label randomization check
    logger.info("Running label randomization check...")
    model_class = LogisticRegressionModel if model_name == "logreg" else MLPModel

    label_rand = label_randomization_check(
        client_datasets=client_datasets,
        model_class=lambda n: model_class(n),
        x_eval=x_eval,
        background=background,
        shap_values_original=shap_orig,
        n_features=n_features,
        local_epochs=local_epochs,
        n_coalitions=256,
        seed=seed,
    )
    results["label_randomization"] = label_rand
    logger.info("Label rand: spearman_rho=%.4f, pass=%s",
                label_rand["spearman_rho"], label_rand["pass"])

    out_file = out_dir / f"sanity_alpha{alpha:.2f}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_file)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_sanity(cfg, args.seed, args.alpha, Path(args.results_dir))


if __name__ == "__main__":
    main()
