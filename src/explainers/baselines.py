"""Baseline SHAP methods for comparison with BA-FedSHAP.

All baselines share the same return signature:
    np.ndarray of shape (n_eval, d) -- SHAP values.

Baselines implemented:
1. local_shap              -- SHAP computed independently per client
2. naive_aggregated_shap   -- Average of per-client SHAP values
3. shared_background_shap  -- All clients share a common background set
4. kmeans_background_shap  -- k-means compressed background set
5. gradient_federated      -- GradientSHAP aggregated across clients
6. centralized_oracle_shap -- SHAP on full pooled dataset (oracle upper bound)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .kernelshap_wrapper import compute_gradient_shap, compute_shap, subsample_background

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Local SHAP (per-client, no federation)
# ---------------------------------------------------------------------------

def local_shap(
    client_dataset: Dict,
    model: Any,
    x_eval: Optional[np.ndarray] = None,
    background_size: int = 200,
    n_coalitions: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    """Compute KernelSHAP independently on a single client's local data.

    Args:
        client_dataset: Dict with key 'X' (client features).
        model: Global or client-local model.
        x_eval: Evaluation points. If None, uses client data.
        background_size: K for background subsampling.
        n_coalitions: KernelSHAP coalition samples.
        seed: RNG seed.

    Returns:
        SHAP values, shape (n_eval, d).
    """
    X_local = client_dataset["X"]
    background = subsample_background(X_local, K=background_size, seed=seed)
    if x_eval is None:
        x_eval = X_local

    return compute_shap(
        model=model,
        x_eval=x_eval,
        background=background,
        n_coalitions=n_coalitions,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 2. Naive Aggregated SHAP (average of per-client SHAP values)
# ---------------------------------------------------------------------------

def naive_aggregated_shap(
    client_datasets: List[Dict],
    model: Any,
    x_eval: np.ndarray,
    background_size: int = 200,
    n_coalitions: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    """Average KernelSHAP values computed independently per client.

    Each client uses its own local data as background. Results are
    uniformly averaged (no weighting by client size).
    """
    all_shap = []
    for k, ds in enumerate(client_datasets):
        if ds["n"] < 5:
            logger.warning("Client %d has too few samples (%d); skipping.", k, ds["n"])
            continue
        shap_k = local_shap(
            client_dataset=ds,
            model=model,
            x_eval=x_eval,
            background_size=min(background_size, ds["n"]),
            n_coalitions=n_coalitions,
            seed=seed + k,
        )
        all_shap.append(shap_k)

    if not all_shap:
        raise ValueError("No valid client SHAP values to aggregate.")

    return np.mean(all_shap, axis=0)


# ---------------------------------------------------------------------------
# 3. Shared Background SHAP
# ---------------------------------------------------------------------------

def shared_background_shap(
    client_datasets: List[Dict],
    model: Any,
    x_eval: np.ndarray,
    background_pool: np.ndarray,
    n_coalitions: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    """KernelSHAP where all clients share a pre-constructed background pool.

    The background_pool is typically a random subsample from a held-out
    reference dataset or a mix of client data shared securely.
    """
    shap_values = compute_shap(
        model=model,
        x_eval=x_eval,
        background=background_pool,
        n_coalitions=n_coalitions,
        seed=seed,
    )
    return shap_values


# ---------------------------------------------------------------------------
# 4. k-means Background SHAP
# ---------------------------------------------------------------------------

def kmeans_background_shap(
    client_datasets: List[Dict],
    model: Any,
    x_eval: np.ndarray,
    k: int = 200,
    n_coalitions: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    """KernelSHAP using k-means cluster centres from the pooled client data.

    Reduces communication cost by summarising each client's data as k-means
    centroids. All centroids are then collected server-side.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except ImportError as exc:
        raise ImportError("scikit-learn is required for k-means background.") from exc

    # Aggregate all client data into pooled set
    X_all = np.concatenate([ds["X"] for ds in client_datasets if ds["n"] > 0], axis=0)

    n_clusters = min(k, len(X_all))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
    km.fit(X_all)
    background = km.cluster_centers_.astype(np.float32)

    return compute_shap(
        model=model,
        x_eval=x_eval,
        background=background,
        n_coalitions=n_coalitions,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 5. Gradient Federated SHAP
# ---------------------------------------------------------------------------

def gradient_federated(
    client_datasets: List[Dict],
    model: Any,
    x_eval: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    n_samples: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Federated GradientSHAP: average gradient-based SHAP per client.

    Each client computes GradientSHAP using its own data as baseline;
    server averages the results weighted by client data size.
    """
    all_shap = []
    all_weights = []

    for k, ds in enumerate(client_datasets):
        if ds["n"] < 2:
            continue
        bl = baseline if baseline is not None else subsample_background(ds["X"], K=50, seed=seed)
        shap_k = compute_gradient_shap(
            model=model,
            x_eval=x_eval,
            baseline=bl,
            seed=seed + k,
            n_samples=n_samples,
        )
        all_shap.append(shap_k)
        all_weights.append(float(ds["n"]))

    if not all_shap:
        raise ValueError("No valid client GradientSHAP values.")

    weights = np.array(all_weights) / sum(all_weights)
    return sum(w * s for w, s in zip(weights, all_shap))


# ---------------------------------------------------------------------------
# 6. Centralized Oracle SHAP (upper bound)
# ---------------------------------------------------------------------------

def centralized_oracle_shap(
    pooled_data: np.ndarray,
    model: Any,
    x_eval: np.ndarray,
    background_size: int = 200,
    n_coalitions: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    """KernelSHAP on the full pooled dataset. Acts as the oracle upper bound.

    In a real federated setting this is not achievable; it serves as a
    reference for measuring how well federated methods approximate centralized
    explanations.
    """
    background = subsample_background(pooled_data, K=background_size, seed=seed)
    return compute_shap(
        model=model,
        x_eval=x_eval,
        background=background,
        n_coalitions=n_coalitions,
        seed=seed,
    )
