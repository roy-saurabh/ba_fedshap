#!/usr/bin/env python3
"""Comprehensive evaluation script for BA-FedSHAP — generates all Tables IV-X metrics.

Usage:
    python scripts/run_full_eval.py --config configs/adult.yaml --seed 42 --alpha 0.5 [options]

CPU-friendly overrides:
    --n-clients INT       override n_clients (default: from config)
    --fl-rounds INT       override fl_rounds (default: from config)
    --coalitions INT      override kernelshap_coalitions (default: from config)
    --background-size INT override background_size (default: from config)
    --max-client-eval INT subsample client data for SHAP (default: no limit)
    --n-lipschitz INT     lipschitz pairs (default: 200)

Output: results/raw/{dataset}/seed_{seed}/alpha_{alpha}/full_eval.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_dataset
from src.data.partition import dirichlet_partition
from src.explainers.baselines import (
    centralized_oracle_shap,
    gradient_federated,
    kmeans_background_shap,
    local_shap as local_shap_fn,
    naive_aggregated_shap,
    shared_background_shap,
)
from src.explainers.kernelshap_wrapper import compute_shap, subsample_background
from src.federated.ba_fedshap import BAFedSHAP
from src.federated.training import FederatedTrainer, build_model
from src.metrics.adi import (
    bootstrap_ci,
    compute_adi_norm,
    compute_l1_to_global,
    compute_spearman_rank,
    permutation_pvalue,
)
from src.metrics.faithfulness import faithfulness_report
from src.metrics.lipschitz import estimate_local_lipschitz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# suppress shap verbose logging
logging.getLogger("shap").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BA-FedSHAP full evaluation.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--results-dir", type=str, default="results/raw")
    p.add_argument("--n-clients", type=int, default=None)
    p.add_argument("--fl-rounds", type=int, default=None)
    p.add_argument("--coalitions", type=int, default=None)
    p.add_argument("--background-size", type=int, default=None)
    p.add_argument("--max-client-eval", type=int, default=None)
    p.add_argument("--n-lipschitz", type=int, default=200)
    p.add_argument("--skip-sanity", action="store_true", help="Skip sanity checks (faster)")
    p.add_argument("--skip-lipschitz", action="store_true")
    return p.parse_args()


def _adi_from_shap(shap_vals: np.ndarray, prot_arr: np.ndarray, s_min: float = 0.001) -> float:
    """Compute ADI_norm from SHAP values and binary protected attribute array."""
    groups = np.unique(prot_arr)
    if len(groups) < 2:
        return 0.0
    g0, g1 = sorted(groups)[:2]
    mu0 = shap_vals[prot_arr == g0].mean(axis=0)
    mu1 = shap_vals[prot_arr == g1].mean(axis=0)
    norms = np.array([np.linalg.norm(shap_vals[prot_arr == g].mean(axis=0))
                      for g in groups])
    s_hat = float(np.std(norms)) if len(groups) > 1 else s_min
    s_hat = max(s_hat, s_min)
    diff = np.abs(mu0 - mu1)
    return float(np.sum(diff) / s_hat)


def _bootstrap_ci_from_summaries(
    client_summaries: list,
    B: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
    s_min: float = 0.001,
) -> tuple:
    """Bootstrap CI for protocol ADI by resampling client summaries.

    Each client has group_mus = {group_id: mean_shap_vector}.
    Bootstrap resamples clients → recomputes aggregated group means → ADI.
    """
    rng = np.random.default_rng(seed)
    valid = [cs for cs in client_summaries if cs.get("group_mus")]
    K = len(valid)
    if K < 2:
        return float("nan"), float("nan"), float("nan")

    def _adi_from_summaries(sample):
        group_vecs: dict = {}
        for cs in sample:
            for g, mu in cs["group_mus"].items():
                g = int(g)
                if g not in group_vecs:
                    group_vecs[g] = []
                group_vecs[g].append(np.array(mu))
        groups = sorted(group_vecs.keys())
        if len(groups) < 2:
            return 0.0
        g0, g1 = groups[0], groups[1]
        mu0 = np.mean(group_vecs[g0], axis=0)
        mu1 = np.mean(group_vecs[g1], axis=0)
        all_mus = [np.mean(vecs, axis=0) for vecs in group_vecs.values()]
        norms = [np.linalg.norm(m) for m in all_mus]
        s_hat = max(float(np.std(norms)), s_min)
        return float(np.sum(np.abs(mu0 - mu1)) / s_hat)

    point_est = _adi_from_summaries(valid)
    boot_stats = []
    for _ in range(B):
        idx = rng.integers(0, K, size=K)
        sample = [valid[i] for i in idx]
        try:
            boot_stats.append(_adi_from_summaries(sample))
        except Exception:
            continue

    if not boot_stats:
        return point_est, float("nan"), float("nan")

    boot_arr = np.array(boot_stats)
    ci_lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    return point_est, ci_lo, ci_hi


def _param_rand_cosine(model, x_eval, background, M, seed) -> float:
    """Cosine similarity between original SHAP and SHAP after full parameter randomisation."""
    import torch
    orig_shap = compute_shap(model, x_eval, background, n_coalitions=M, seed=seed)
    # Randomise all parameters
    rand_model = copy.deepcopy(model)
    torch.manual_seed(seed + 77777)
    with torch.no_grad():
        for param in rand_model.parameters():
            param.data = torch.randn_like(param.data)
    rand_shap = compute_shap(rand_model, x_eval, background, n_coalitions=M, seed=seed)
    # Mean SHAP vectors
    orig_mu = orig_shap.mean(axis=0)
    rand_mu = rand_shap.mean(axis=0)
    cos = float(np.dot(orig_mu, rand_mu) / (np.linalg.norm(orig_mu) * np.linalg.norm(rand_mu) + 1e-12))
    return float(np.abs(cos))


def _label_rand_spearman(X, y, sensitive, cfg, seed, alpha, M, K_global, background,
                         x_eval, orig_global_mu, n_clients, fl_rounds, model_name) -> float:
    """Spearman correlation between original SHAP ranks and SHAP from label-randomised model."""
    rng_lr = np.random.default_rng(seed + 88888)
    y_rand = rng_lr.permutation(y)
    sensitive_rand = sensitive  # keep same
    client_ds_rand = dirichlet_partition(X, y_rand, sensitive_rand, n_clients, alpha, seed)
    rand_model = build_model(model_name, X.shape[1])
    trainer_rand = FederatedTrainer(seed=seed + 88888)
    rand_model = trainer_rand.fit(client_ds_rand, rand_model, rounds=fl_rounds,
                                  local_epochs=cfg.get("local_epochs", 5),
                                  lr=cfg.get("lr", 0.01), momentum=cfg.get("momentum", 0.9))
    rand_shap = compute_shap(rand_model, x_eval, background, n_coalitions=M, seed=seed)
    rand_mu = rand_shap.mean(axis=0)
    rho, _ = stats.spearmanr(np.abs(orig_global_mu), np.abs(rand_mu))
    return float(rho)


def run_full_eval(cfg: dict, seed: int, alpha: float, results_dir: Path,
                  n_clients_ov=None, fl_rounds_ov=None, coalitions_ov=None,
                  background_ov=None, max_client_eval=None,
                  n_lipschitz=200, skip_sanity=False, skip_lipschitz=False) -> None:

    t_start = time.time()

    dataset_name = cfg["dataset"]
    n_clients = n_clients_ov or cfg.get("n_clients", 50)
    fl_rounds = fl_rounds_ov or cfg.get("fl_rounds", 200)
    local_epochs = cfg.get("local_epochs", 5)
    lr = cfg.get("lr", 0.01)
    momentum = cfg.get("momentum", 0.9)
    K_global = background_ov or cfg.get("background_size", 200)
    M = coalitions_ov or cfg.get("kernelshap_coalitions", 2048)
    dp_sigmas = cfg.get("dp_sigmas", {"inf": 0.0})
    model_name = cfg.get("model", "logreg")
    C = cfg.get("dp_clip_c", 1.0)
    z_tau = cfg.get("drift_z_tau", 3.5)
    s_min = 0.001
    B_boot = cfg.get("bootstrap_b", 200)   # reduced for speed
    B_perm = cfg.get("permutation_bp", 200)

    out_dir = results_dir / dataset_name / f"seed_{seed}" / f"alpha_{alpha:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Full Eval: %s | seed=%d | alpha=%.2f ===", dataset_name, seed, alpha)
    logger.info("    n_clients=%d, fl_rounds=%d, M=%d, K=%d", n_clients, fl_rounds, M, K_global)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    kwargs = {}
    if dataset_name in ("acs_income", "acs_pubcov"):
        kwargs["survey_year"] = cfg.get("acs_survey_year", "2018")
        kwargs["states"] = cfg.get("acs_states", None)
    X, y, sensitive = load_dataset(dataset_name, **kwargs)
    n_features = X.shape[1]
    prot_key = list(sensitive.keys())[0]
    prot_arr = sensitive[prot_key]

    # ------------------------------------------------------------------
    # 2. Partition
    # ------------------------------------------------------------------
    client_datasets = dirichlet_partition(X, y, sensitive, n_clients, alpha, seed)
    # Drop clients that got 0 samples (can happen with small datasets + low alpha)
    client_datasets = [d for d in client_datasets if d["n"] > 0]

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    t_train = time.time()
    global_model = build_model(model_name, n_features)
    trainer = FederatedTrainer(seed=seed)
    global_model = trainer.fit(
        client_datasets, global_model, rounds=fl_rounds,
        local_epochs=local_epochs, lr=lr, momentum=momentum,
    )
    eval_metrics = trainer.evaluate(global_model, client_datasets)
    train_time_s = time.time() - t_train
    logger.info("Model trained: acc=%.4f  (%.1fs)", eval_metrics["accuracy_mean"], train_time_s)

    # ------------------------------------------------------------------
    # 4. Reference / background / eval sets
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    n_ref = max(K_global, int(0.1 * len(y)))
    ref_idx = rng.choice(len(y), size=n_ref, replace=False)
    ref_data = X[ref_idx]
    ref_prot = prot_arr[ref_idx]
    background_global = subsample_background(ref_data, K=K_global, seed=seed)

    n_eval = min(300, len(y) - n_ref)
    remaining = np.setdiff1d(np.arange(len(y)), ref_idx)
    eval_idx = rng.choice(remaining, size=n_eval, replace=False)
    x_eval = X[eval_idx]
    y_eval = y[eval_idx]
    prot_eval = prot_arr[eval_idx]

    # Subsample client datasets for SHAP
    client_datasets_shap = client_datasets
    if max_client_eval is not None:
        rng_sub = np.random.default_rng(seed + 9999)
        client_datasets_shap = []
        for ds in client_datasets:
            if ds["n"] > max_client_eval:
                idx_s = rng_sub.choice(ds["n"], size=max_client_eval, replace=False)
                client_datasets_shap.append({
                    "client_id": ds.get("client_id"),
                    "X": ds["X"][idx_s],
                    "y": ds["y"][idx_s],
                    "sensitive": {k: v[idx_s] for k, v in ds["sensitive"].items()},
                    "n": max_client_eval,
                })
            else:
                client_datasets_shap.append(ds)

    # ------------------------------------------------------------------
    # 5. Oracle: centralized SHAP (reference for all comparisons)
    # ------------------------------------------------------------------
    logger.info("Computing oracle SHAP...")
    t0 = time.time()
    oracle_shap = centralized_oracle_shap(X, global_model, x_eval, K_global, M, seed)
    oracle_global_mu = oracle_shap.mean(axis=0)
    oracle_adi = _adi_from_shap(oracle_shap, prot_eval, s_min)
    oracle_runtime = time.time() - t0
    logger.info("Oracle ADI=%.4f (%.1fs)", oracle_adi, oracle_runtime)

    # ------------------------------------------------------------------
    # 6. BA-FedSHAP protocol (all DP levels)
    # ------------------------------------------------------------------
    ba_results: dict = {}
    protocol = BAFedSHAP(K_global=K_global, n_coalitions=M, seed=seed)

    for eps_label, sigma in dp_sigmas.items():
        logger.info("BA-FedSHAP eps=%s sigma=%.4f ...", eps_label, sigma)
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
            proto = protocol.run_full_protocol(
                client_datasets=client_datasets_shap,
                global_model=global_model,
                ref_data=ref_data,
                ref_protected=ref_prot,
                config=run_cfg,
            )
            rt = time.time() - t0

            global_mu = np.array(proto["global_mu"])
            client_mus = [
                np.array(list(cs["group_mus"].values())).mean(axis=0)
                for cs in proto["client_summaries"]
                if cs["group_mus"]
            ]
            prot_labels_clients = np.array([
                list(cs["group_mus"].keys())[0]
                for cs in proto["client_summaries"]
                if cs["group_mus"]
            ])

            method_adi = proto["adi_norm"]
            adi_err = abs(method_adi - oracle_adi)

            # Primary metrics: compare aggregated global_mu to oracle (same as baselines)
            l1_global = float(np.sum(np.abs(global_mu - oracle_global_mu)))
            rho_global, _ = stats.spearmanr(np.abs(global_mu), np.abs(oracle_global_mu))

            # Per-client metrics (secondary, used for CI)
            rho_mean, rho_std = compute_spearman_rank(client_mus, oracle_global_mu)

            # Bootstrap CI and oracle coverage (resample client summaries)
            if len(proto["client_summaries"]) >= 2:
                pt, ci_lo, ci_hi = _bootstrap_ci_from_summaries(
                    proto["client_summaries"], B=B_boot, seed=seed, s_min=s_min
                )
                ci_covers = 1.0 if (not np.isnan(ci_lo) and ci_lo <= oracle_adi <= ci_hi) else 0.0
            else:
                ci_lo, ci_hi, ci_covers = float("nan"), float("nan"), float("nan")

            # Faithfulness using protocol's global background
            q_pool = background_global  # use global background for consistent comparison
            try:
                ba_shap = compute_shap(global_model, x_eval, q_pool,
                                       n_coalitions=min(M, 128), seed=seed)
                faith_ba = faithfulness_report(global_model, x_eval, ba_shap, background_global)
                del_auc_ba = faith_ba["deletion_auc"]
                ins_auc_ba = faith_ba["insertion_auc"]
            except Exception:
                del_auc_ba = float("nan")
                ins_auc_ba = float("nan")

            ba_results[str(eps_label)] = {
                "adi_norm": float(method_adi),
                "adi_oracle": float(oracle_adi),
                "adi_error": float(adi_err),
                "l1_to_global_mean": l1_global,
                "spearman_rho_mean": float(rho_global),
                "spearman_rho_std": float(rho_std),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "ci_covers_oracle": float(ci_covers),
                "n_clients_used": proto["n_clients_used"],
                "outlier_clients": proto["outlier_clients"],
                "runtime_s": rt,
                "deletion_auc": del_auc_ba,
                "insertion_auc": ins_auc_ba,
            }
            logger.info(
                "  eps=%s: ADI=%.4f oracle=%.4f err=%.4f L1=%.4f rho=%.4f (%.1fs)",
                eps_label, method_adi, oracle_adi, adi_err, l1_global, rho_global, rt,
            )
        except Exception as exc:
            logger.error("BA-FedSHAP eps=%s FAILED: %s", eps_label, exc, exc_info=True)
            ba_results[str(eps_label)] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 7. Baselines
    # ------------------------------------------------------------------
    baseline_results: dict = {}

    def _run_baseline(name, shap_fn, **kw):
        logger.info("Baseline: %s ...", name)
        t0 = time.time()
        try:
            shap_vals = shap_fn(**kw)
            rt = time.time() - t0
            mu = shap_vals.mean(axis=0)
            method_adi = _adi_from_shap(shap_vals, prot_eval, s_min)
            adi_err = abs(method_adi - oracle_adi)
            l1 = float(np.sum(np.abs(mu - oracle_global_mu)))
            rho, _ = stats.spearmanr(np.abs(mu), np.abs(oracle_global_mu))
            faith = faithfulness_report(global_model, x_eval, shap_vals, background_global)
            baseline_results[name] = {
                "adi_norm": float(method_adi),
                "adi_oracle": float(oracle_adi),
                "adi_error": float(adi_err),
                "l1_to_global_mean": float(l1),
                "spearman_rho_mean": float(rho),
                "deletion_auc": faith["deletion_auc"],
                "insertion_auc": faith["insertion_auc"],
                "runtime_s": rt,
                "shap_mean_abs_norm": float(np.abs(shap_vals).mean()),
            }
            logger.info(
                "  %s: ADI=%.4f err=%.4f del=%.4f ins=%.4f (%.1fs)",
                name, method_adi, adi_err,
                faith["deletion_auc"], faith["insertion_auc"], rt,
            )
        except Exception as exc:
            logger.error("Baseline %s FAILED: %s", name, exc)
            baseline_results[name] = {"error": str(exc)}

    _run_baseline(
        "centralized_oracle",
        centralized_oracle_shap,
        pooled_data=X, model=global_model, x_eval=x_eval,
        background_size=K_global, n_coalitions=M, seed=seed,
    )
    def _local_shap_pooled():
        # Each client computes SHAP with local background; pool all shap values
        shap_list = []
        for ds in client_datasets_shap:
            if ds["n"] < 5:
                continue
            sv = local_shap_fn(ds, global_model, x_eval=x_eval,
                               background_size=min(K_global, ds["n"]),
                               n_coalitions=M, seed=seed)
            shap_list.append(sv)
        if not shap_list:
            return np.zeros((len(x_eval), X.shape[1]), dtype=np.float32)
        return np.mean(shap_list, axis=0)

    _run_baseline("local_shap", _local_shap_pooled)
    _run_baseline(
        "naive_aggregated",
        naive_aggregated_shap,
        client_datasets=client_datasets_shap, model=global_model, x_eval=x_eval,
        background_size=K_global, n_coalitions=M, seed=seed,
    )
    _run_baseline(
        "shared_background",
        shared_background_shap,
        client_datasets=client_datasets_shap, model=global_model, x_eval=x_eval,
        background_pool=background_global, n_coalitions=M, seed=seed,
    )
    _run_baseline(
        "kmeans_background",
        kmeans_background_shap,
        client_datasets=client_datasets_shap, model=global_model, x_eval=x_eval,
        k=K_global, n_coalitions=M, seed=seed,
    )

    # BA-FedSHAP faithfulness (inf DP)
    if "inf" in ba_results and "error" not in ba_results["inf"]:
        logger.info("Computing BA-FedSHAP faithfulness...")
        try:
            backgrounds_strat = protocol.construct_backgrounds(ref_data, ref_prot, K_global, seed)
            q_pool = backgrounds_strat["__global__"]
            baf_shap = compute_shap(global_model, x_eval, q_pool, n_coalitions=M, seed=seed)
            faith_baf = faithfulness_report(global_model, x_eval, baf_shap, background_global)
            ba_results["inf"]["deletion_auc"] = faith_baf["deletion_auc"]
            ba_results["inf"]["insertion_auc"] = faith_baf["insertion_auc"]
        except Exception as exc:
            logger.error("BA-FedSHAP faithfulness failed: %s", exc)

    # ------------------------------------------------------------------
    # 8. Sanity checks (optional)
    # ------------------------------------------------------------------
    sanity = {}
    if not skip_sanity:
        logger.info("Running sanity checks...")
        try:
            cos = _param_rand_cosine(global_model, x_eval, background_global, M, seed)
            sanity["param_rand_cosine"] = cos
            logger.info("  Param rand cosine: %.4f", cos)
        except Exception as exc:
            logger.error("Param rand cosine failed: %s", exc)
            sanity["param_rand_cosine"] = float("nan")

        try:
            orig_mu = oracle_shap.mean(axis=0)
            rho_lr = _label_rand_spearman(
                X, y, sensitive, cfg, seed, alpha, M, K_global,
                background_global, x_eval, orig_mu, n_clients, fl_rounds, model_name
            )
            sanity["label_rand_spearman"] = rho_lr
            logger.info("  Label rand Spearman: %.4f", rho_lr)
        except Exception as exc:
            logger.error("Label rand Spearman failed: %s", exc)
            sanity["label_rand_spearman"] = float("nan")

    # ------------------------------------------------------------------
    # 9. Lipschitz estimation (optional)
    # ------------------------------------------------------------------
    lipschitz = {}
    if not skip_lipschitz:
        logger.info("Estimating Lipschitz constant (n_pairs=%d)...", n_lipschitz)
        try:
            lip = estimate_local_lipschitz(
                global_model, x_eval, background_global,
                n_pairs=n_lipschitz, eps_norm=0.01, n_coalitions=min(M, 64), seed=seed,
            )
            lipschitz = lip
            logger.info("  L_hat=%.4f mean=%.4f", lip["l_hat"], lip["l_hat_mean"])
        except Exception as exc:
            logger.error("Lipschitz estimation failed: %s", exc)
            lipschitz = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 10. Save all results
    # ------------------------------------------------------------------
    total_time = time.time() - t_start

    results = {
        "dataset": dataset_name,
        "seed": seed,
        "alpha": alpha,
        "n_features": n_features,
        "n_clients": n_clients,
        "fl_rounds": fl_rounds,
        "kernelshap_coalitions": M,
        "background_size": K_global,
        "eval_accuracy": eval_metrics["accuracy_mean"],
        "eval_accuracy_std": eval_metrics["accuracy_std"],
        "oracle_adi": float(oracle_adi),
        "oracle_runtime_s": oracle_runtime,
        "ba_fedshap": ba_results,
        "baselines": baseline_results,
        "sanity": sanity,
        "lipschitz": lipschitz,
        "total_runtime_s": total_time,
    }

    out_file = out_dir / "full_eval.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s (total %.1fs)", out_file, total_time)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_full_eval(
        cfg=cfg,
        seed=args.seed,
        alpha=args.alpha,
        results_dir=Path(args.results_dir),
        n_clients_ov=args.n_clients,
        fl_rounds_ov=args.fl_rounds,
        coalitions_ov=args.coalitions,
        background_ov=args.background_size,
        max_client_eval=args.max_client_eval,
        n_lipschitz=args.n_lipschitz,
        skip_sanity=args.skip_sanity,
        skip_lipschitz=args.skip_lipschitz,
    )


if __name__ == "__main__":
    main()
