"""Figure generation utilities for BA-FedSHAP results.

Provides functions to generate the main paper figures from results files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import matplotlib to avoid issues in headless environments
def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Figure 1: ADI_norm vs epsilon (DP noise sweep)
# ---------------------------------------------------------------------------

def plot_adi_vs_epsilon(
    results_by_epsilon: Dict[str, float],
    dataset_name: str = "adult",
    save_path: Optional[Path] = None,
) -> Path:
    """Plot ADI_norm as a function of DP epsilon budget.

    Args:
        results_by_epsilon: Dict mapping epsilon label (str) to ADI_norm scalar.
        dataset_name: Dataset name for title.
        save_path: If given, saves figure there. Otherwise saves to results/figures/.

    Returns:
        Path where figure was saved.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 4))

    epsilons = list(results_by_epsilon.keys())
    adis = [results_by_epsilon[e] for e in epsilons]

    ax.plot(range(len(epsilons)), adis, marker="o", linewidth=2)
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels(epsilons)
    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("ADI_norm")
    ax.set_title(f"ADI_norm vs ε — {dataset_name}")
    ax.grid(True, alpha=0.3)

    out_path = save_path or (
        Path("results/figures") / f"adi_vs_epsilon_{dataset_name}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Reliability (bootstrap CI plot)
# ---------------------------------------------------------------------------

def plot_reliability_ci(
    method_names: List[str],
    adi_means: List[float],
    adi_ci_lo: List[float],
    adi_ci_hi: List[float],
    dataset_name: str = "adult",
    save_path: Optional[Path] = None,
) -> Path:
    """Plot ADI_norm with bootstrap confidence intervals per method.

    Args:
        method_names: List of method labels.
        adi_means: Point estimates of ADI_norm.
        adi_ci_lo: Lower CI bounds.
        adi_ci_hi: Upper CI bounds.
        dataset_name: Dataset name.
        save_path: Output path.

    Returns:
        Path where figure was saved.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(method_names))
    yerr_lo = np.array(adi_means) - np.array(adi_ci_lo)
    yerr_hi = np.array(adi_ci_hi) - np.array(adi_means)

    ax.bar(x, adi_means, yerr=[yerr_lo, yerr_hi], capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.set_ylabel("ADI_norm")
    ax.set_title(f"ADI_norm (95% CI) — {dataset_name}")
    ax.grid(True, axis="y", alpha=0.3)

    out_path = save_path or (
        Path("results/figures") / f"reliability_ci_{dataset_name}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: SHAP feature importance bar plot
# ---------------------------------------------------------------------------

def plot_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    title: str = "Global SHAP Feature Importance",
    top_k: int = 20,
    save_path: Optional[Path] = None,
) -> Path:
    """Horizontal bar chart of mean absolute SHAP values.

    Args:
        shap_values: SHAP values, shape (N, d).
        feature_names: List of feature names, length d.
        title: Plot title.
        top_k: Number of top features to show.
        save_path: Output path.

    Returns:
        Path where figure was saved.
    """
    plt = _get_plt()

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in order]
    values = mean_abs[order]

    ax.barh(range(len(order)), values[::-1], alpha=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("|SHAP value| (mean)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    out_path = save_path or Path("results/figures/shap_importance.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: Faithfulness curves
# ---------------------------------------------------------------------------

def plot_faithfulness_curves(
    method_names: List[str],
    deletion_aucs: List[float],
    insertion_aucs: List[float],
    dataset_name: str = "adult",
    save_path: Optional[Path] = None,
) -> Path:
    """Grouped bar chart for Deletion and Insertion AUC per method."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(method_names))
    width = 0.35
    ax.bar(x - width / 2, deletion_aucs, width, label="Deletion AUC", alpha=0.7)
    ax.bar(x + width / 2, insertion_aucs, width, label="Insertion AUC", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title(f"Faithfulness Metrics — {dataset_name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out_path = save_path or (
        Path("results/figures") / f"faithfulness_{dataset_name}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Placeholder figure generator
# ---------------------------------------------------------------------------

def generate_placeholder_figures(figures_dir: Path = Path("results/figures")) -> None:
    """Generate empty placeholder PNGs for all expected figure files.

    Used when experiment results don't exist yet, to allow the paper pipeline
    to run without errors.
    """
    plt = _get_plt()
    figures_dir.mkdir(parents=True, exist_ok=True)

    placeholders = [
        "adi_vs_epsilon_adult.png",
        "adi_vs_epsilon_compas.png",
        "reliability_ci_adult.png",
        "reliability_ci_compas.png",
        "faithfulness_adult.png",
        "shap_importance.png",
        "lipschitz_adult.png",
        "ablation_adult.png",
        "dp_sweep_adult.png",
        "stability_adult.png",
    ]

    for fname in placeholders:
        fpath = figures_dir / fname
        if not fpath.exists():
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, f"[Placeholder]\n{fname}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_axis_off()
            fig.savefig(fpath, dpi=72)
            plt.close(fig)
            logger.debug("Created placeholder: %s", fpath)

    logger.info("Placeholder figures generated in %s", figures_dir)
