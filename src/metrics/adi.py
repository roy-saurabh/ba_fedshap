"""ADI_norm (Attribution Disparity Index, normalised) metric for BA-FedSHAP.

ADI_norm measures the disparity between SHAP-based feature attributions across
demographic groups. A higher ADI_norm indicates larger group-level attribution
differences.

All public functions accept numpy arrays. Bootstrapping and permutation
testing assume B = 2000 resamples by default, matching the paper.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardised Mean Difference (SMD)
# ---------------------------------------------------------------------------

def compute_smd(
    mu_global_a: np.ndarray,
    mu_global_aprime: np.ndarray,
    s_hat: float,
    s_min: float = 0.001,
) -> float:
    """Compute the Standardised Mean Difference (SMD) between two group means.

    SMD = ||mu_a - mu_a'||_1 / max(s_hat, s_min)

    Args:
        mu_global_a: Mean SHAP vector for group a, shape (d,).
        mu_global_aprime: Mean SHAP vector for group a', shape (d,).
        s_hat: Estimated pooled standard deviation.
        s_min: Minimum denominator to avoid division by zero.

    Returns:
        Scalar SMD value >= 0.
    """
    diff = np.abs(mu_global_a - mu_global_aprime)
    denom = max(float(s_hat), s_min)
    return float(np.sum(diff) / denom)


# ---------------------------------------------------------------------------
# ADI_norm
# ---------------------------------------------------------------------------

def compute_adi_norm(
    client_mus: List[np.ndarray],
    protected_vals: np.ndarray,
    s_min: float = 0.001,
) -> float:
    """Compute ADI_norm across all clients.

    For each client k, client_mus[k] is the mean SHAP vector for that client.
    protected_vals[k] is the protected group label for client k (binary: 0 or 1).

    ADI_norm = SMD(mu_global_a=0, mu_global_a=1) / max(s_hat, s_min)

    where:
        mu_global_a = mean over clients with protected_vals == a
        s_hat = std of per-client mean SHAP norms

    Args:
        client_mus: List of mean SHAP vectors, one per client.
        protected_vals: Binary group label per client (0 or 1).
        s_min: Minimum standard deviation floor.

    Returns:
        ADI_norm scalar.
    """
    client_mus_arr = np.array(client_mus)  # shape (K, d)
    protected_vals = np.array(protected_vals)

    groups = np.unique(protected_vals)
    if len(groups) < 2:
        logger.warning("Only one protected group found; ADI_norm = 0.")
        return 0.0

    mu_by_group = {}
    for g in groups:
        mask = protected_vals == g
        mu_by_group[g] = client_mus_arr[mask].mean(axis=0)

    # Pooled norm-based std
    norms = np.linalg.norm(client_mus_arr, axis=1)
    s_hat = float(np.std(norms)) if len(norms) > 1 else s_min

    # Use first two groups
    g0, g1 = sorted(groups)[:2]
    return compute_smd(mu_by_group[g0], mu_by_group[g1], s_hat=s_hat, s_min=s_min)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for ADI_norm
# ---------------------------------------------------------------------------

def bootstrap_ci(
    client_mus: List[np.ndarray],
    protected_vals: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap (B=2000) confidence interval for ADI_norm.

    Args:
        client_mus: Per-client mean SHAP vectors.
        protected_vals: Binary group label per client.
        B: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 → 95% CI).
        seed: RNG seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    K = len(client_mus)
    protected_vals = np.array(protected_vals)

    point_est = compute_adi_norm(client_mus, protected_vals)

    boot_stats = []
    for _ in range(B):
        idx = rng.choice(K, size=K, replace=True)
        boot_mus = [client_mus[i] for i in idx]
        boot_prot = protected_vals[idx]
        try:
            boot_stats.append(compute_adi_norm(boot_mus, boot_prot))
        except Exception:
            continue

    boot_stats = np.array(boot_stats)
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point_est, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Permutation p-value for ADI_norm
# ---------------------------------------------------------------------------

def permutation_pvalue(
    client_mus: List[np.ndarray],
    protected_vals: np.ndarray,
    B_p: int = 2000,
    seed: int = 42,
) -> float:
    """Permutation test p-value for ADI_norm.

    Null hypothesis: protected group label is unrelated to SHAP attribution.

    Args:
        client_mus: Per-client mean SHAP vectors.
        protected_vals: Binary group label per client.
        B_p: Number of permutation resamples.
        seed: RNG seed.

    Returns:
        Two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    protected_vals = np.array(protected_vals)

    observed = compute_adi_norm(client_mus, protected_vals)
    perm_stats = []

    for _ in range(B_p):
        perm_prot = rng.permutation(protected_vals)
        try:
            perm_stats.append(compute_adi_norm(client_mus, perm_prot))
        except Exception:
            continue

    perm_stats = np.array(perm_stats)
    p_val = float(np.mean(perm_stats >= observed))
    return p_val


# ---------------------------------------------------------------------------
# L1 distance to global mean
# ---------------------------------------------------------------------------

def compute_l1_to_global(
    client_mus: List[np.ndarray],
    global_mu: np.ndarray,
) -> np.ndarray:
    """Per-client L1 distance of mean SHAP vector to global mean.

    Args:
        client_mus: List of mean SHAP vectors per client.
        global_mu: Global mean SHAP vector (shape d,).

    Returns:
        Array of L1 distances, shape (K,).
    """
    return np.array([np.sum(np.abs(mu - global_mu)) for mu in client_mus])


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------

def compute_spearman_rank(
    client_mus: List[np.ndarray],
    global_mu: np.ndarray,
) -> Tuple[float, float]:
    """Mean cross-client Spearman rank correlation of per-client SHAP rankings
    with the global SHAP ranking.

    Args:
        client_mus: Per-client mean SHAP vectors (shape d, each).
        global_mu: Global mean SHAP vector (shape d,).

    Returns:
        (mean_rho, std_rho) across clients.
    """
    rhos = []
    global_rank = np.argsort(np.abs(global_mu))[::-1]
    for mu in client_mus:
        client_rank = np.argsort(np.abs(mu))[::-1]
        rho, _ = stats.spearmanr(global_rank, client_rank)
        rhos.append(float(rho))

    return float(np.mean(rhos)), float(np.std(rhos))
