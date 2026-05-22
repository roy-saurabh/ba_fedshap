#!/usr/bin/env python3
"""Run ablation study (Table VI) on Adult dataset.

Ablation variants:
  1. full: BA-FedSHAP with all components
  2. no_drift: BA-FedSHAP without drift detection (z_tau=inf)
  3. no_stratified_bg: BA-FedSHAP with global background only (no per-group stratification)
  4. with_dp_eps4: BA-FedSHAP with DP eps=4 (sigma=0.1583)
  5. with_dp_eps1: BA-FedSHAP with DP eps=1 (sigma=0.5942)

Results saved to results/raw/{dataset}/ablation/
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
from src.explainers.kernelshap_wrapper import compute_shap, subsample_background
from src.federated.ba_fedshap import BAFedSHAP
from src.federated.training import FederatedTrainer, build_model
from src.metrics.adi import bootstrap_ci, compute_adi_norm
from src.metrics.faithfulness import faithfulness_report

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BA-FedSHAP ablation study.")
    p.add_argument("--config", type=str, default="configs/adult.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--results-dir", type=str, default="results/raw")
    p.add_argument("--n-clients", type=int, default=None)
    p.add_argument("--fl-rounds", type=int, default=None)
    p.add_argument("--coalitions", type=int, default=None)
    p.add_argument("--background-size", type=int, default=None)
    p.add_argument("--max-client-eval", type=int, default=None)
    return p.parse_args()


# Corrected sigma values (Theorem 2 RDP accountant, T=200 rounds, q=0.1)
ABLATION_VARIANTS = {
    "full": {"use_drift": True, "use_stratified_bg": True, "sigma": 0.0},
    "no_drift": {"use_drift": False, "use_stratified_bg": True, "sigma": 0.0},
    "no_stratified_bg": {"use_drift": True, "use_stratified_bg": False, "sigma": 0.0},
    "with_dp_eps4": {"use_drift": True, "use_stratified_bg": True, "sigma": 0.1583},
    "with_dp_eps1": {"use_drift": True, "use_stratified_bg": True, "sigma": 0.5942},
}


def run_ablation(cfg: dict, seed: int, alpha: float, results_dir: Path,
                 n_clients_ov=None, fl_rounds_ov=None, coalitions_ov=None,
                 background_ov=None, max_client_eval=None) -> None:
    dataset_name = cfg["dataset"]
    n_clients = n_clients_ov or cfg.get("n_clients", 50)
    fl_rounds = fl_rounds_ov or cfg.get("fl_rounds", 200)
    local_epochs = cfg.get("local_epochs", 5)
    K_global = background_ov or cfg.get("background_size", 200)
    M = coalitions_ov or cfg.get("kernelshap_coalitions", 2048)
    model_name = cfg.get("model", "logreg")
    C = cfg.get("dp_clip_c", 1.0)

    out_dir = results_dir / dataset_name / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if dataset_name in ("acs_income", "acs_pubcov"):
        kwargs["survey_year"] = cfg.get("acs_survey_year", "2018")
        kwargs["states"] = cfg.get("acs_states", None)

    X, y, sensitive = load_dataset(dataset_name, **kwargs)
    n_features = X.shape[1]
    rng = np.random.default_rng(seed)

    client_datasets = dirichlet_partition(X, y, sensitive, n_clients, alpha, seed)
    client_datasets = [d for d in client_datasets if d["n"] > 0]

    # Subsample large clients for SHAP speed
    if max_client_eval is not None:
        rng_sub = np.random.default_rng(seed + 9999)
        trimmed = []
        for ds in client_datasets:
            if ds["n"] > max_client_eval:
                idx = rng_sub.choice(ds["n"], size=max_client_eval, replace=False)
                trimmed.append({
                    "X": ds["X"][idx], "y": ds["y"][idx],
                    "sensitive": {k: v[idx] for k, v in ds["sensitive"].items()},
                    "n": max_client_eval, "client_id": ds["client_id"],
                })
            else:
                trimmed.append(ds)
        client_datasets = trimmed

    model = build_model(model_name, n_features)
    trainer = FederatedTrainer(seed=seed)
    model = trainer.fit(client_datasets, model, rounds=fl_rounds, local_epochs=local_epochs,
                        lr=cfg.get("lr", 0.01), momentum=cfg.get("momentum", 0.9))

    ref_idx = rng.choice(len(y), size=min(K_global, len(y)), replace=False)
    ref_data = X[ref_idx].astype(np.float32)
    prot_key = list(sensitive.keys())[0]
    ref_prot = sensitive[prot_key][ref_idx]

    eval_idx = rng.choice(len(y), size=min(200, len(y)), replace=False)
    x_eval = X[eval_idx]
    background_global = subsample_background(ref_data, K=K_global, seed=seed)

    results = {
        "dataset": dataset_name,
        "seed": seed,
        "alpha": alpha,
        "n_clients": n_clients,
        "fl_rounds": fl_rounds,
        "ablation": {},
    }

    protocol = BAFedSHAP(K_global=K_global, n_coalitions=M, seed=seed)

    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        logger.info("Ablation variant: %s", variant_name)
        t0 = time.time()
        try:
            z_tau = 3.5 if variant_cfg["use_drift"] else float("inf")
            sigma = variant_cfg["sigma"]

            if variant_cfg["use_stratified_bg"]:
                backgrounds = protocol.construct_backgrounds(ref_data, ref_prot, K_global, seed)
            else:
                backgrounds = {"__global__": background_global}

            q_pool = backgrounds.get("__global__", background_global)

            client_summaries = []
            for k, ds in enumerate(client_datasets):
                if ds["n"] == 0:
                    continue
                cs = protocol.client_compute(
                    client_data=ds,
                    model=model,
                    q_pool=q_pool,
                    q_a_dict=backgrounds,
                    C=C,
                    sigma_dp=sigma,
                    seed=seed + k,
                )
                client_summaries.append(cs)

            agg = protocol.server_aggregate(
                client_summaries=client_summaries,
                z_tau=z_tau,
            )

            runtime_s = time.time() - t0

            global_shap = compute_shap(model, x_eval, q_pool, n_coalitions=min(M, 128), seed=seed)
            faith = faithfulness_report(model, x_eval, global_shap, background_global)

            results["ablation"][variant_name] = {
                "adi_norm": agg["adi_norm"],
                "n_clients_used": agg["n_clients_used"],
                "deletion_auc": faith["deletion_auc"],
                "insertion_auc": faith["insertion_auc"],
                "runtime_s": runtime_s,
            }
            logger.info(
                "%s: ADI=%.4f, del_auc=%.4f, ins_auc=%.4f, time=%.1fs",
                variant_name, agg["adi_norm"],
                faith["deletion_auc"], faith["insertion_auc"], runtime_s,
            )

        except Exception as exc:
            logger.error("Ablation %s failed: %s", variant_name, exc, exc_info=True)
            results["ablation"][variant_name] = {"error": str(exc)}

    out_file = out_dir / f"ablation_seed{seed}_alpha{alpha:.2f}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_file)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_ablation(
        cfg, args.seed, args.alpha, Path(args.results_dir),
        n_clients_ov=args.n_clients,
        fl_rounds_ov=args.fl_rounds,
        coalitions_ov=args.coalitions,
        background_ov=args.background_size,
        max_client_eval=args.max_client_eval,
    )


if __name__ == "__main__":
    main()
