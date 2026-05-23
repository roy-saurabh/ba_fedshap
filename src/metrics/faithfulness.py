"""Faithfulness metrics for SHAP explanations: Deletion AUC and Insertion AUC.

These metrics measure whether SHAP values correctly identify the most important
features. Deletion AUC should decrease quickly (good); Insertion AUC should
increase quickly (good).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _trapz_compat(values: list, dx: float) -> float:
    """Trapezoidal integration compatible with NumPy < 2.0 and >= 2.0."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, dx=dx)
    return np.trapz(values, dx=dx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_features(
    x: np.ndarray,
    feature_order: np.ndarray,
    n_replaced: int,
    background: np.ndarray,
    mode: str = "deletion",
) -> np.ndarray:
    """Return a modified copy of x with n_replaced features replaced.

    In deletion mode: the n_replaced most important features are replaced by
    their background mean (features are masked out).
    In insertion mode: all features EXCEPT the n_replaced most important are
    replaced by background mean (only top-k features are kept).
    """
    bg_mean = background.mean(axis=0)
    x_mod = np.full_like(x, fill_value=bg_mean)

    if mode == "deletion":
        # Keep all features EXCEPT the top n_replaced
        keep_idx = feature_order[n_replaced:]
    else:  # insertion
        # Keep only the top n_replaced features
        keep_idx = feature_order[:n_replaced]

    x_mod[:, keep_idx] = x[:, keep_idx]
    return x_mod


# ---------------------------------------------------------------------------
# Deletion AUC
# ---------------------------------------------------------------------------

def deletion_auc(
    model: Any,
    x: np.ndarray,
    shap_values: np.ndarray,
    background: np.ndarray,
    n_steps: int = 10,
) -> float:
    """Compute Deletion AUC.

    Features are removed in descending order of |SHAP value| (most important
    first). At each step, removed features are replaced by their background
    mean. The model's predicted probability is recorded. Lower AUC is better
    (explanations correctly identify features the model relies on).

    Args:
        model: Any object with a `predict_proba(X)` method returning (N, 2).
        x: Evaluation samples, shape (N, d).
        shap_values: SHAP values, shape (N, d).
        background: Background data for mean imputation, shape (K, d).
        n_steps: Number of deletion steps (evenly spaced between 0 and d).

    Returns:
        Scalar AUC (mean over samples).
    """
    N, d = x.shape
    # Feature importance order: descending absolute SHAP (averaged over samples)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_order = np.argsort(mean_abs_shap)[::-1]

    step_sizes = np.linspace(0, d, n_steps + 1, dtype=int)
    scores = []

    for n_removed in step_sizes:
        x_mod = _replace_features(x, feature_order, int(n_removed), background, mode="deletion")
        proba = model.predict_proba(x_mod)[:, 1]
        scores.append(float(proba.mean()))

    # AUC via trapezoidal rule, normalised by range
    auc = float(_trapz_compat(scores, dx=1.0 / n_steps))
    logger.debug("Deletion AUC = %.4f", auc)
    return auc


# ---------------------------------------------------------------------------
# Insertion AUC
# ---------------------------------------------------------------------------

def insertion_auc(
    model: Any,
    x: np.ndarray,
    shap_values: np.ndarray,
    background: np.ndarray,
    n_steps: int = 10,
) -> float:
    """Compute Insertion AUC.

    Features are added back in descending order of |SHAP value|. At each step,
    the model's predicted probability is recorded. Higher AUC is better.

    Args:
        model: Any object with a `predict_proba(X)` method.
        x: Evaluation samples, shape (N, d).
        shap_values: SHAP values, shape (N, d).
        background: Background data for mean imputation, shape (K, d).
        n_steps: Number of insertion steps.

    Returns:
        Scalar AUC (mean over samples).
    """
    N, d = x.shape
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_order = np.argsort(mean_abs_shap)[::-1]

    step_sizes = np.linspace(0, d, n_steps + 1, dtype=int)
    scores = []

    for n_inserted in step_sizes:
        x_mod = _replace_features(x, feature_order, int(n_inserted), background, mode="insertion")
        proba = model.predict_proba(x_mod)[:, 1]
        scores.append(float(proba.mean()))

    auc = float(_trapz_compat(scores, dx=1.0 / n_steps))
    logger.debug("Insertion AUC = %.4f", auc)
    return auc


# ---------------------------------------------------------------------------
# Combined faithfulness report
# ---------------------------------------------------------------------------

def faithfulness_report(
    model: Any,
    x: np.ndarray,
    shap_values: np.ndarray,
    background: np.ndarray,
    n_steps: int = 10,
) -> dict:
    """Return both Deletion and Insertion AUC in a dict."""
    del_auc = deletion_auc(model, x, shap_values, background, n_steps=n_steps)
    ins_auc = insertion_auc(model, x, shap_values, background, n_steps=n_steps)
    return {
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "faithfulness_gap": ins_auc - del_auc,  # larger gap = more faithful
    }
