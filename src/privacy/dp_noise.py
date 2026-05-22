"""Differential Privacy noise mechanisms for BA-FedSHAP.

Provides:
- clip_l2: L2-norm clipping (sensitivity bounding)
- add_gaussian_noise: Gaussian noise injection
- noisy_mean: Per-vector clipping + mean + Gaussian noise
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L2 clipping
# ---------------------------------------------------------------------------

def clip_l2(
    vector: np.ndarray,
    C: float = 1.0,
) -> np.ndarray:
    """Clip a vector to have at most L2 norm C.

    If ||v||_2 <= C, returns v unchanged.
    Otherwise returns v * C / ||v||_2.

    Args:
        vector: 1D or 2D array. If 2D (N, d), clips each row independently.
        C: Clipping threshold.

    Returns:
        Clipped array of same shape.
    """
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")

    if vector.ndim == 1:
        norm = np.linalg.norm(vector)
        if norm > C:
            return vector * (C / norm)
        return vector.copy()

    elif vector.ndim == 2:
        norms = np.linalg.norm(vector, axis=1, keepdims=True)  # (N, 1)
        scale = np.minimum(1.0, C / np.maximum(norms, 1e-12))
        return vector * scale

    else:
        raise ValueError(f"vector must be 1D or 2D, got shape {vector.shape}")


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    vector: np.ndarray,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """Add isotropic Gaussian noise N(0, sigma^2 * I) to a vector.

    Args:
        vector: Input array (any shape).
        sigma: Standard deviation of noise.
        seed: Optional RNG seed.

    Returns:
        Noisy array of same shape.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return vector.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=vector.shape).astype(vector.dtype)
    return vector + noise


# ---------------------------------------------------------------------------
# Noisy mean
# ---------------------------------------------------------------------------

def noisy_mean(
    vectors: list[np.ndarray] | np.ndarray,
    C: float = 1.0,
    sigma: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Compute differentially private mean:
    1. Clip each vector to L2 norm C.
    2. Sum clipped vectors.
    3. Divide by N (number of vectors).
    4. Add Gaussian noise N(0, (sigma * C / N)^2 * I).

    This implements the standard DP mean mechanism used in FL.

    Args:
        vectors: List of 1D arrays or 2D array (N, d).
        C: L2 clipping constant.
        sigma: Noise multiplier (relative to sensitivity C/N).
        seed: RNG seed.

    Returns:
        Noisy mean vector, shape (d,).
    """
    if isinstance(vectors, list):
        vectors_arr = np.array(vectors, dtype=np.float32)
    else:
        vectors_arr = np.array(vectors, dtype=np.float32)

    N, d = vectors_arr.shape

    # Step 1: Clip each row
    clipped = clip_l2(vectors_arr, C=C)

    # Step 2 & 3: Mean
    mean_vec = clipped.mean(axis=0)

    # Step 4: Add noise with effective std = sigma * C / N
    if sigma > 0 and N > 0:
        effective_sigma = sigma * C / N
        mean_vec = add_gaussian_noise(mean_vec, sigma=effective_sigma, seed=seed)

    return mean_vec
