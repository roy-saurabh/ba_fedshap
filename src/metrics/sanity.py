"""Sanity checks for SHAP explanations.

Two checks are implemented:
1. Parameter randomization check: randomise model weights -> SHAP should change.
2. Label randomization check: train on shuffled labels -> SHAP should be near-zero.

Both follow the methodology of Adebayo et al. (2018) adapted for FL+SHAP.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, List, Optional, Type

import numpy as np
from scipy import stats  # type: ignore

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

from ..explainers.kernelshap_wrapper import compute_shap, subsample_background

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter randomization check
# ---------------------------------------------------------------------------

def parameter_randomization_check(
    model: Any,
    x_eval: np.ndarray,
    background: np.ndarray,
    shap_values_original: np.ndarray,
    n_coalitions: int = 512,
    seed: int = 42,
) -> dict:
    """Randomize model weights and re-compute SHAP. A good explanation method
    should produce significantly different (low cosine similarity) SHAP values
    after randomization.

    Args:
        model: Trained model.
        x_eval: Evaluation samples, shape (n_eval, d).
        background: Background samples, shape (K, d).
        shap_values_original: Original SHAP values, shape (n_eval, d).
        n_coalitions: Number of coalitions for randomized SHAP (fewer = faster).
        seed: RNG seed.

    Returns:
        Dict with 'cosine_similarity' (lower = better sanity).
    """
    torch.manual_seed(seed)
    randomized_model = copy.deepcopy(model)

    # Randomize all parameters
    with torch.no_grad():
        for param in randomized_model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.1)

    shap_rand = compute_shap(
        model=randomized_model,
        x_eval=x_eval,
        background=background,
        n_coalitions=n_coalitions,
        seed=seed,
    )

    # Compute cosine similarity between flattened SHAP vectors
    orig_flat = shap_values_original.flatten()
    rand_flat = shap_rand.flatten()
    norm_o = np.linalg.norm(orig_flat)
    norm_r = np.linalg.norm(rand_flat)

    if norm_o < 1e-12 or norm_r < 1e-12:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(orig_flat, rand_flat) / (norm_o * norm_r))

    logger.info("Parameter randomization cosine similarity: %.4f", cos_sim)
    return {
        "cosine_similarity": cos_sim,
        "shap_original_norm": float(norm_o),
        "shap_randomized_norm": float(norm_r),
        "pass": cos_sim < 0.5,  # expect low similarity after randomization
    }


# ---------------------------------------------------------------------------
# Label randomization check
# ---------------------------------------------------------------------------

def label_randomization_check(
    client_datasets: List[dict],
    model_class: Any,
    x_eval: np.ndarray,
    background: np.ndarray,
    shap_values_original: np.ndarray,
    n_features: int,
    local_epochs: int = 5,
    lr: float = 0.01,
    momentum: float = 0.9,
    n_coalitions: int = 512,
    seed: int = 42,
) -> dict:
    """Train model on randomized labels; SHAP values should be near-zero.

    Args:
        client_datasets: List of client data dicts.
        model_class: Model class constructor (takes n_features).
        x_eval: Evaluation samples.
        background: Background samples.
        shap_values_original: Original SHAP values.
        n_features: Feature dimensionality.
        local_epochs: Epochs for local training on random labels.
        lr: Learning rate.
        momentum: SGD momentum.
        n_coalitions: KernelSHAP coalition samples.
        seed: RNG seed.

    Returns:
        Dict with 'spearman_rho' (expected near 0 after label randomization).
    """
    from ..federated.training import FederatedTrainer, fedavg_aggregate, local_train

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Create randomized-label datasets
    rand_datasets = []
    for ds in client_datasets:
        rand_ds = dict(ds)
        rand_ds["y"] = rng.integers(0, 2, size=len(ds["y"])).astype(ds["y"].dtype)
        rand_datasets.append(rand_ds)

    # Train model on randomized labels for a few rounds
    init_model = model_class(n_features)
    trainer = FederatedTrainer(seed=seed)
    rand_model = trainer.fit(
        client_datasets=rand_datasets,
        global_model=init_model,
        rounds=10,  # small number for sanity check
        local_epochs=local_epochs,
        lr=lr,
        momentum=momentum,
    )

    shap_rand = compute_shap(
        model=rand_model,
        x_eval=x_eval,
        background=background,
        n_coalitions=n_coalitions,
        seed=seed,
    )

    # Compute Spearman correlation of mean absolute SHAP vectors
    mean_abs_orig = np.abs(shap_values_original).mean(axis=0)
    mean_abs_rand = np.abs(shap_rand).mean(axis=0)

    rho, p_val = stats.spearmanr(mean_abs_orig, mean_abs_rand)
    logger.info("Label randomization Spearman rho: %.4f (p=%.4f)", rho, p_val)

    return {
        "spearman_rho": float(rho),
        "spearman_pvalue": float(p_val),
        "pass": abs(rho) < 0.3,  # expect near-zero correlation
    }
