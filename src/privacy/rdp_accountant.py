"""Rényi Differential Privacy (RDP) accountant for BA-FedSHAP.

Implements the RDP composition theorem for the Gaussian mechanism,
as stated in Theorem 2 of the paper.

epsilon_rdp(alpha) = alpha * sensitivity^2 / (2 * sigma^2)

This is composed T times (one per FL round), then converted to (epsilon, delta)-DP
via the standard RDP-to-DP conversion.

Reference:
    Mironov (2017). Rényi Differential Privacy.
    Balle et al. (2020). Hypothesis Testing Interpretations and Renyi DP.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default alpha orders to evaluate
_DEFAULT_ALPHAS = list(range(2, 65)) + [128, 256, 512, 1024]


# ---------------------------------------------------------------------------
# Per-round RDP epsilon
# ---------------------------------------------------------------------------

def rdp_per_round(
    alpha: float,
    sensitivity_delta2: float,
    sigma: float,
) -> float:
    """RDP epsilon for a single Gaussian mechanism application.

    From Theorem 2 (Gaussian mechanism):
        epsilon_rdp(alpha) = alpha * sensitivity^2 / (2 * sigma^2)

    Args:
        alpha: Rényi order (alpha > 1).
        sensitivity_delta2: L2 sensitivity of the mechanism (delta_2 in paper,
                            equals the clipping constant C divided by number of
                            participating clients: C / m_k^a).
        sigma: Noise multiplier (standard deviation of Gaussian noise
               relative to sensitivity).

    Returns:
        RDP epsilon for this round.
    """
    if sigma <= 0:
        return float("inf")
    return float(alpha * (sensitivity_delta2 ** 2) / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Composition over T rounds
# ---------------------------------------------------------------------------

def rdp_compose(
    alpha: float,
    epsilon_per_round: float,
    T: int,
) -> float:
    """Compose RDP epsilon over T rounds of adaptive composition.

    Under RDP, epsilon_rdp composes linearly:
        epsilon_rdp_total(alpha) = T * epsilon_rdp_per_round(alpha)

    Args:
        alpha: Rényi order.
        epsilon_per_round: RDP epsilon for one round.
        T: Number of rounds.

    Returns:
        Total composed RDP epsilon.
    """
    return float(T * epsilon_per_round)


# ---------------------------------------------------------------------------
# RDP-to-(epsilon, delta)-DP conversion
# ---------------------------------------------------------------------------

def rdp_to_dp(
    alpha: float,
    epsilon_rdp: float,
    delta: float,
) -> float:
    """Convert (alpha, epsilon_rdp) to (epsilon, delta)-DP.

    Using the tight conversion of Balle et al. (2020):
        epsilon_dp = epsilon_rdp + log(1 - 1/alpha) - log(delta) / (alpha - 1)
                   (simplified / approximate form)

    The paper uses the standard form from Mironov (2017):
        epsilon_dp = epsilon_rdp - log(delta) / (alpha - 1)

    Args:
        alpha: Rényi order (> 1).
        epsilon_rdp: Composed RDP epsilon.
        delta: Target delta.

    Returns:
        Standard (epsilon, delta)-DP epsilon.
    """
    if alpha <= 1:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if math.isinf(epsilon_rdp):
        return float("inf")

    eps_dp = epsilon_rdp + math.log(1.0 - 1.0 / alpha) - math.log(delta) / (alpha - 1.0)
    return float(eps_dp)


# ---------------------------------------------------------------------------
# Compute total (epsilon, delta)-DP over all alpha orders
# ---------------------------------------------------------------------------

def compute_epsilon(
    T: int,
    sigma: float,
    sensitivity_delta2: float,
    delta: float,
    alpha_range: Optional[Sequence[float]] = None,
) -> float:
    """Compute the tightest (epsilon, delta)-DP guarantee over a range of alpha orders.

    Evaluates RDP over multiple alpha values and returns the minimum epsilon.

    Args:
        T: Total number of FL rounds.
        sigma: Gaussian noise multiplier.
        sensitivity_delta2: L2 sensitivity per round.
        delta: Target delta.
        alpha_range: Rényi orders to evaluate. Defaults to [2..64, 128, 256, 512, 1024].

    Returns:
        Minimum (tightest) epsilon_dp over all alpha values.
    """
    if sigma <= 0:
        return float("inf")

    alphas = alpha_range if alpha_range is not None else _DEFAULT_ALPHAS
    best_eps = float("inf")

    for alpha in alphas:
        try:
            eps_rdp_round = rdp_per_round(alpha, sensitivity_delta2, sigma)
            eps_rdp_total = rdp_compose(alpha, eps_rdp_round, T)
            eps_dp = rdp_to_dp(alpha, eps_rdp_total, delta)
            best_eps = min(best_eps, eps_dp)
        except Exception:
            continue

    return best_eps


# ---------------------------------------------------------------------------
# Compute sigma for a target epsilon
# ---------------------------------------------------------------------------

def compute_sigma_for_epsilon(
    target_epsilon: float,
    T: int,
    sensitivity_delta2: float,
    delta: float,
    C: float = 1.0,
    m_k_a: int = 100,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> float:
    """Binary search for sigma achieving target (epsilon, delta)-DP.

    Args:
        target_epsilon: Desired (epsilon, delta)-DP epsilon budget.
        T: Number of FL rounds.
        sensitivity_delta2: L2 sensitivity (C / m_k_a if not specified separately).
        delta: Target delta.
        C: Clipping constant (used to parameterise sensitivity if sensitivity_delta2
           is provided directly, this is ignored).
        m_k_a: Approximate number of clients per group (for sensitivity scaling).
        tol: Convergence tolerance.
        max_iter: Maximum binary search iterations.

    Returns:
        Sigma value achieving approximately target_epsilon.
    """
    if math.isinf(target_epsilon):
        return 0.0  # No noise needed for epsilon=inf (no DP)

    lo, hi = 1e-6, 1000.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        eps_mid = compute_epsilon(T, mid, sensitivity_delta2, delta)
        if abs(eps_mid - target_epsilon) < tol:
            return float(mid)
        if eps_mid > target_epsilon:
            lo = mid
        else:
            hi = mid

    return float((lo + hi) / 2.0)


# ---------------------------------------------------------------------------
# Convenience: compute epsilon for paper's standard settings
# ---------------------------------------------------------------------------

def paper_epsilon_table(
    T: int = 200,
    C: float = 1.0,
    n_clients: int = 50,
    delta: float = 1e-5,
    sigma_map: Optional[dict] = None,
) -> dict:
    """Compute epsilon for each sigma in the paper's Table VII noise settings.

    Args:
        T: FL rounds.
        C: Clipping constant.
        n_clients: Number of clients (used to compute per-round sensitivity).
        delta: Target delta.
        sigma_map: Dict mapping epsilon_label to sigma. Defaults to paper values.

    Returns:
        Dict mapping sigma_label to computed (epsilon, delta)-DP epsilon.
    """
    if sigma_map is None:
        sigma_map = {
            "inf": 0.0,   # no noise
            "8": 0.018,
            "4": 0.035,
            "2": 0.071,
            "1": 0.141,
        }

    sensitivity_delta2 = C / n_clients
    results = {}

    for label, sigma in sigma_map.items():
        if sigma <= 0:
            eps = float("inf")
        else:
            eps = compute_epsilon(T, sigma, sensitivity_delta2, delta)
        results[label] = eps
        logger.info("sigma=%s -> epsilon=%.4f (delta=%.1e)", label, eps, delta)

    return results
