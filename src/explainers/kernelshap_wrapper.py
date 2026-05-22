"""KernelSHAP and GradientSHAP wrappers for BA-FedSHAP.

Provides a unified interface for computing SHAP values from PyTorch models.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KernelSHAP
# ---------------------------------------------------------------------------

def compute_shap(
    model: Any,
    x_eval: np.ndarray,
    background: np.ndarray,
    n_coalitions: int = 2048,
    seed: int = 42,
    link: str = "identity",
) -> np.ndarray:
    """Compute KernelSHAP values for `x_eval` using `background` as reference.

    Args:
        model: PyTorch model with `.predict_proba(X)` returning shape (N, 2).
        x_eval: Evaluation samples, shape (n_eval, d).
        background: Background/reference samples, shape (K, d).
        n_coalitions: Number of feature coalitions to sample (nsamples in shap).
        seed: Random seed for reproducibility.
        link: Link function ('identity' or 'logit').

    Returns:
        SHAP values array of shape (n_eval, d).
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError("shap>=0.44.0 is required: pip install shap==0.44.0") from exc

    np.random.seed(seed)

    def _predict(X: np.ndarray) -> np.ndarray:
        """Return P(Y=1) for each row of X."""
        return model.predict_proba(X)[:, 1]

    logger.debug(
        "KernelSHAP: n_eval=%d, K=%d, coalitions=%d",
        len(x_eval), len(background), n_coalitions,
    )

    explainer = shap.KernelExplainer(model.predict_proba, background, link=link)
    shap_values = explainer.shap_values(x_eval, nsamples=n_coalitions, seed=seed)

    # shap_values from KernelExplainer on binary classification:
    # shap>=0.46: returns ndarray of shape (n_eval, d, n_classes)
    # shap<0.46: returns list [shap_class0, shap_class1] where each is (n_eval, d)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (n_eval, d, n_classes) — take positive class
        shap_values = shap_values[:, :, 1]

    return np.array(shap_values, dtype=np.float32)


# ---------------------------------------------------------------------------
# GradientSHAP
# ---------------------------------------------------------------------------

def compute_gradient_shap(
    model: Any,
    x_eval: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    seed: int = 42,
    n_samples: int = 50,
) -> np.ndarray:
    """Compute GradientSHAP values using shap.GradientExplainer.

    Args:
        model: PyTorch model (must have a `forward` method).
        x_eval: Evaluation samples, shape (n_eval, d).
        baseline: Baseline/background samples, shape (K, d). If None, uses
                  zero tensor of same shape as x_eval.
        seed: Random seed.
        n_samples: Number of random samples for gradient estimation.

    Returns:
        SHAP values array of shape (n_eval, d).
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError("shap>=0.44.0 is required.") from exc

    np.random.seed(seed)
    torch.manual_seed(seed)

    if baseline is None:
        baseline = np.zeros_like(x_eval[:1])

    # GradientExplainer expects torch tensors
    x_tensor = torch.tensor(x_eval, dtype=torch.float32)
    bg_tensor = torch.tensor(baseline, dtype=torch.float32)

    explainer = shap.GradientExplainer(model, bg_tensor)
    shap_values = explainer.shap_values(x_tensor, nsamples=n_samples, ranked_outputs=None)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().numpy()

    return np.array(shap_values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Utility: subsample background
# ---------------------------------------------------------------------------

def subsample_background(
    data: np.ndarray,
    K: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Randomly subsample K rows from data for use as KernelSHAP background."""
    rng = np.random.default_rng(seed)
    if len(data) <= K:
        return data
    idx = rng.choice(len(data), size=K, replace=False)
    return data[idx]
