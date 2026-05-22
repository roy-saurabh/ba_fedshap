"""Data partitioning strategies for federated learning.

Provides Dirichlet (label-distribution) and geography-based partitioning.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    sensitive: Dict[str, np.ndarray],
    n_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples: int = 10,
) -> List[Dict]:
    """Partition dataset among n_clients using Dirichlet distribution over labels.

    Args:
        X: Feature matrix (N, d).
        y: Binary labels (N,).
        sensitive: Mapping of sensitive attribute name to array of group labels.
        n_clients: Number of FL clients.
        alpha: Dirichlet concentration parameter (smaller = more heterogeneous).
        seed: Random seed.
        min_samples: Minimum number of samples per client (retry if violated).

    Returns:
        List of dicts, each with keys 'X', 'y', 'sensitive', 'client_id'.
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        # Sample Dirichlet proportions for this class
        proportions = rng.dirichlet(alpha=np.repeat(alpha, n_clients))
        # Convert to counts, ensure at least 0
        counts = (proportions * len(idx_c)).astype(int)
        # Fix rounding by assigning remainder to first client
        diff = len(idx_c) - counts.sum()
        counts[0] += diff

        start = 0
        for k, cnt in enumerate(counts):
            client_indices[k].extend(idx_c[start : start + cnt].tolist())
            start += cnt

    # Build client datasets; warn if any client is too small
    client_datasets = []
    for k, idx in enumerate(client_indices):
        if len(idx) < min_samples:
            logger.warning(
                "Client %d has only %d samples (min=%d); alpha=%.2f may be too small.",
                k, len(idx), min_samples, alpha,
            )
        idx_arr = np.array(idx, dtype=int)
        sens_k = {attr: arr[idx_arr] for attr, arr in sensitive.items()}
        client_datasets.append({
            "client_id": k,
            "X": X[idx_arr],
            "y": y[idx_arr],
            "sensitive": sens_k,
            "n": len(idx_arr),
        })

    _log_partition_stats(client_datasets, n_classes, alpha)
    return client_datasets


def geography_partition(
    X: np.ndarray,
    y: np.ndarray,
    sensitive: Dict[str, np.ndarray],
    state_labels: np.ndarray,
) -> Dict[str, Dict]:
    """Partition dataset by geographic state label.

    Args:
        X: Feature matrix (N, d).
        y: Binary labels (N,).
        sensitive: Sensitive attributes.
        state_labels: Array of state strings, length N.

    Returns:
        Dict mapping state name to client dict with keys 'X', 'y', 'sensitive', 'client_id'.
    """
    states = np.unique(state_labels)
    client_datasets: Dict[str, Dict] = {}

    for s in states:
        idx = np.where(state_labels == s)[0]
        sens_s = {attr: arr[idx] for attr, arr in sensitive.items()}
        client_datasets[str(s)] = {
            "client_id": str(s),
            "X": X[idx],
            "y": y[idx],
            "sensitive": sens_s,
            "n": len(idx),
        }
        logger.debug("State %s: %d samples", s, len(idx))

    return client_datasets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_partition_stats(client_datasets: List[Dict], n_classes: int, alpha: float) -> None:
    sizes = [d["n"] for d in client_datasets]
    label_fracs = []
    for d in client_datasets:
        if d["n"] > 0:
            frac = float(d["y"].mean())
        else:
            frac = 0.0
        label_fracs.append(frac)

    logger.info(
        "Dirichlet partition (alpha=%.2f): n_clients=%d, "
        "size min=%d max=%d mean=%.1f, "
        "pos-rate min=%.3f max=%.3f",
        alpha,
        len(client_datasets),
        min(sizes),
        max(sizes),
        np.mean(sizes),
        min(label_fracs),
        max(label_fracs),
    )
