"""Lipschitz constant estimation for model explanation stability.

Estimates the local Lipschitz constant L̂ of the SHAP explanation function
via random perturbation pairs. A lower L̂ indicates more stable explanations
under small input perturbations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..explainers.kernelshap_wrapper import compute_shap

logger = logging.getLogger(__name__)


def estimate_local_lipschitz(
    model: Any,
    X_sample: np.ndarray,
    background: np.ndarray,
    n_pairs: int = 1000,
    eps_norm: float = 0.01,
    n_coalitions: int = 128,
    seed: int = 42,
) -> dict:
    """Estimate local Lipschitz constant L̂ over random perturbation pairs.

    For each pair (x, x'), where x' = x + eps * u (u unit vector):
        L̂ = max over pairs of ||phi(x) - phi(x')||_2 / ||x - x'||_2

    where phi(x) is the SHAP explanation vector for x.

    Args:
        model: Trained model with `predict_proba`.
        X_sample: Evaluation samples, shape (N, d). A random subset of n_pairs
                  samples is drawn.
        background: Background data for KernelSHAP, shape (K, d).
        n_pairs: Number of (x, x') pairs to evaluate.
        eps_norm: L2 norm of the perturbation (determines neighbourhood size).
        n_coalitions: KernelSHAP coalitions (fewer for speed).
        seed: RNG seed.

    Returns:
        Dict with:
          'l_hat': Estimated local Lipschitz constant.
          'l_hat_mean': Mean Lipschitz ratio over all pairs.
          'l_hat_p95': 95th percentile.
          'n_pairs_evaluated': Actual number of pairs used.
    """
    rng = np.random.default_rng(seed)
    N, d = X_sample.shape

    # Sample base points
    n_base = min(n_pairs, N)
    idx = rng.choice(N, size=n_base, replace=False)
    X_base = X_sample[idx]

    # Generate perturbations: random unit vectors scaled by eps_norm
    U = rng.normal(0.0, 1.0, size=(n_base, d)).astype(np.float32)
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norms, 1e-12) * eps_norm
    X_perturbed = (X_base + U).astype(np.float32)

    # Clip perturbed samples to [0, 1] (valid feature range after min-max scaling)
    X_perturbed = np.clip(X_perturbed, 0.0, 1.0)

    logger.info(
        "Estimating Lipschitz: %d pairs, eps_norm=%.4f, K=%d",
        n_base, eps_norm, len(background),
    )

    # Compute SHAP for base and perturbed in batches
    # Use small n_coalitions for speed
    batch_size = min(50, n_base)
    lipschitz_ratios = []

    for i in range(0, n_base, batch_size):
        bsl = slice(i, min(i + batch_size, n_base))
        x_b = X_base[bsl]
        x_p = X_perturbed[bsl]

        local_seed_b = int(rng.integers(0, 2**31))
        local_seed_p = int(rng.integers(0, 2**31))

        shap_base = compute_shap(
            model, x_b, background, n_coalitions=n_coalitions, seed=local_seed_b
        )
        shap_pert = compute_shap(
            model, x_p, background, n_coalitions=n_coalitions, seed=local_seed_p
        )

        # Per-sample Lipschitz ratio
        delta_shap = np.linalg.norm(shap_base - shap_pert, axis=1)  # (batch,)
        delta_x = np.linalg.norm(x_b - x_p, axis=1)                 # (batch,)

        # Avoid division by zero
        valid = delta_x > 1e-12
        if valid.any():
            ratios = delta_shap[valid] / delta_x[valid]
            lipschitz_ratios.extend(ratios.tolist())

    if not lipschitz_ratios:
        logger.warning("No valid Lipschitz pairs found.")
        return {"l_hat": float("nan"), "l_hat_mean": float("nan"),
                "l_hat_p95": float("nan"), "n_pairs_evaluated": 0}

    ratios_arr = np.array(lipschitz_ratios)
    result = {
        "l_hat": float(ratios_arr.max()),
        "l_hat_mean": float(ratios_arr.mean()),
        "l_hat_p95": float(np.percentile(ratios_arr, 95)),
        "n_pairs_evaluated": len(ratios_arr),
    }
    logger.info("L̂ = %.4f (mean=%.4f, p95=%.4f)", result["l_hat"],
                result["l_hat_mean"], result["l_hat_p95"])
    return result
