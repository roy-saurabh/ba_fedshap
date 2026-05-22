#!/usr/bin/env python3
"""Sweep DP epsilon in {1, 2, 4, 8, inf} for BA-FedSHAP on a given dataset.

Results saved to results/raw/{dataset}/dp_sweep/
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_dataset
from src.data.partition import dirichlet_partition
from src.federated.ba_fedshap import BAFedSHAP
from src.federated.training import FederatedTrainer, build_model
from src.metrics.adi import bootstrap_ci

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BA-FedSHAP DP epsilon sweep.")
    p.add_argument("--config", type=str, default="configs/adult.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--results-dir", type=str, default="results/raw")
    return p.parse_args()


def run_dp_sweep(cfg: dict, seed: int, alpha: float, results_dir: Path) -> None:
    dataset_name = cfg["dataset"]
    n_clients = cfg.get("n_clients", 50)
    fl_rounds = cfg.get("fl_rounds", 200)
    local_epochs = cfg.get("local_epochs", 5)
    K_global = cfg.get("background_size", 200)
    M = cfg.get("kernelshap_coalitions", 2048)
    model_name = cfg.get("model", "logreg")
    C = cfg.get("dp_clip_c", 1.0)
    z_tau = cfg.get("drift_z_tau", 3.5)
    dp_sigmas = cfg.get("dp_sigmas", {"inf": 0.0, "8": 0.018, "4": 0.035, "2": 0.071, "1": 0.141})

    out_dir = results_dir / dataset_name / "dp_sweep"
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
    ref_data = X[ref_idx].astype(np.float32)
    prot_key = list(sensitive.keys())[0]
    ref_prot = sensitive[prot_key][ref_idx]

    protocol = BAFedSHAP(K_global=K_global, n_coalitions=M, seed=seed)
    results = {
        "dataset": dataset_name,
        "seed": seed,
        "alpha": alpha,
        "dp_sweep": {},
    }

    for eps_label, sigma in dp_sigmas.items():
        logger.info("epsilon=%s (sigma=%.4f)", eps_label, sigma)
        run_cfg = {
            "background_size": K_global,
            "dp_clip_c": C,
            "dp_sigma": float(sigma),
            "drift_z_tau": z_tau,
            "kernelshap_coalitions": M,
            "seed": seed,
        }

        t0 = time.time()
        try:
            proto_result = protocol.run_full_protocol(
                client_datasets=client_datasets,
                global_model=model,
                ref_data=ref_data,
                ref_protected=ref_prot,
                config=run_cfg,
            )
            runtime_s = time.time() - t0

            # Bootstrap CI for ADI_norm
            client_mus = [
                np.array(list(cs["group_mus"].values())).mean(axis=0)
                for cs in proto_result["client_summaries"]
                if cs["group_mus"]
            ]
            prot_labels = np.zeros(len(client_mus), dtype=int)
            if len(client_mus) >= 2:
                n_half = len(client_mus) // 2
                prot_labels[n_half:] = 1

            if len(client_mus) >= 2:
                adi_pt, ci_lo, ci_hi = bootstrap_ci(
                    client_mus, prot_labels, B=cfg.get("bootstrap_b", 2000), seed=seed
                )
            else:
                adi_pt, ci_lo, ci_hi = proto_result["adi_norm"], float("nan"), float("nan")

            results["dp_sweep"][str(eps_label)] = {
                "sigma": float(sigma),
                "adi_norm": proto_result["adi_norm"],
                "adi_norm_ci_lo": ci_lo,
                "adi_norm_ci_hi": ci_hi,
                "n_clients_used": proto_result["n_clients_used"],
                "runtime_s": runtime_s,
            }
            logger.info(
                "eps=%s: ADI=%.4f [%.4f, %.4f], time=%.1fs",
                eps_label, adi_pt, ci_lo, ci_hi, runtime_s,
            )
        except Exception as exc:
            logger.error("Failed eps=%s: %s", eps_label, exc)
            results["dp_sweep"][str(eps_label)] = {"error": str(exc)}

    out_file = out_dir / f"dp_sweep_seed{seed}_alpha{alpha:.2f}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_file)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_dp_sweep(cfg, args.seed, args.alpha, Path(args.results_dir))


if __name__ == "__main__":
    main()
