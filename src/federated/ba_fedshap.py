"""BA-FedSHAP: Bias-Aware Federated SHAP protocol.

Implements the three core algorithms from the paper:
  - Algorithm 1: Client-side SHAP computation with DP
  - Algorithm 2: Server-side bias-aware aggregation
  - Algorithm 3: Bias-stratified background construction

The full pipeline is orchestrated by BAFedSHAP.run_full_protocol().
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..explainers.kernelshap_wrapper import compute_shap, subsample_background
from ..privacy.dp_noise import clip_l2, noisy_mean

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BAFedSHAP class
# ---------------------------------------------------------------------------

class BAFedSHAP:
    """Bias-Aware Federated SHAP protocol.

    Attributes:
        K_global: Total background set size.
        n_coalitions: KernelSHAP coalitions per client call.
        seed: Master random seed.
    """

    def __init__(
        self,
        K_global: int = 200,
        n_coalitions: int = 2048,
        seed: int = 42,
    ) -> None:
        self.K_global = K_global
        self.n_coalitions = n_coalitions
        self.seed = seed

    # -----------------------------------------------------------------------
    # Algorithm 3: Construct bias-stratified background
    # -----------------------------------------------------------------------

    def construct_backgrounds(
        self,
        ref_data: np.ndarray,
        protected_col: np.ndarray,
        K_global: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[Any, np.ndarray]:
        """Algorithm 3: Construct bias-stratified background sets Q_a for each group a.

        For each unique value of protected_col, sample K_a = K_global // n_groups
        rows from ref_data restricted to that group.

        Args:
            ref_data: Reference data, shape (N_ref, d).
            protected_col: Protected attribute values for ref_data, shape (N_ref,).
            K_global: Total background size across all groups. Defaults to self.K_global.
            seed: RNG seed. Defaults to self.seed.

        Returns:
            Dict mapping group label -> background array Q_a of shape (K_a, d).
            Also includes key '__global__' with the union of all Q_a.
        """
        rng = np.random.default_rng(seed or self.seed)
        K = K_global or self.K_global
        groups = np.unique(protected_col)
        n_groups = len(groups)
        K_per_group = max(1, K // n_groups)

        backgrounds: Dict[Any, np.ndarray] = {}
        all_bg = []

        for g in groups:
            idx = np.where(protected_col == g)[0]
            k_g = min(K_per_group, len(idx))
            sampled = rng.choice(idx, size=k_g, replace=False)
            bg_g = ref_data[sampled].astype(np.float32)
            backgrounds[g] = bg_g
            all_bg.append(bg_g)
            logger.debug("Background group=%s: K_a=%d", g, k_g)

        backgrounds["__global__"] = np.concatenate(all_bg, axis=0)
        return backgrounds

    # -----------------------------------------------------------------------
    # Algorithm 1: Client-side SHAP computation with DP
    # -----------------------------------------------------------------------

    def client_compute(
        self,
        client_data: Dict,
        model: Any,
        q_pool: np.ndarray,
        q_a_dict: Dict[Any, np.ndarray],
        C: float = 1.0,
        sigma_dp: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict:
        """Algorithm 1: Per-client SHAP computation and DP aggregation.

        For each protected group a present in the client's data:
          1. Select background Q_a.
          2. Compute KernelSHAP for client samples in group a.
          3. Compute per-group mean SHAP vector mu_k^a.
          4. Clip mu_k^a to L2 norm C.
          5. Add DP noise if sigma_dp > 0.

        Args:
            client_data: Dict with 'X', 'y', 'sensitive' (dict of attr->array).
            model: Global model.
            q_pool: Global background pool (fallback).
            q_a_dict: Per-group background sets from construct_backgrounds().
            C: L2 clipping constant.
            sigma_dp: Gaussian noise standard deviation (0 = no DP).
            seed: RNG seed.

        Returns:
            Dict with:
              'client_id': client identifier,
              'group_mus': {group_val: noisy clipped mean SHAP vector},
              'group_sizes': {group_val: n_samples in group},
              'n': total samples,
        """
        _seed = seed or self.seed
        rng = np.random.default_rng(_seed)
        X = client_data["X"]
        n = len(X)

        if n == 0:
            return {
                "client_id": client_data.get("client_id", -1),
                "group_mus": {},
                "group_sizes": {},
                "n": 0,
            }

        # Determine protected attribute to use (first available key)
        sensitive = client_data.get("sensitive", {})
        if not sensitive:
            # No sensitive attribute info; use global background
            shap_all = compute_shap(
                model=model, x_eval=X, background=q_pool,
                n_coalitions=self.n_coalitions, seed=_seed,
            )
            mu = clip_l2(shap_all.mean(axis=0), C=C)
            if sigma_dp > 0:
                from ..privacy.dp_noise import add_gaussian_noise
                mu = add_gaussian_noise(mu, sigma_dp * C, seed=int(rng.integers(0, 2**31)))
            return {
                "client_id": client_data.get("client_id", -1),
                "group_mus": {"__all__": mu},
                "group_sizes": {"__all__": n},
                "n": n,
            }

        # Use first protected attribute
        prot_attr = list(sensitive.keys())[0]
        prot_vals = sensitive[prot_attr]
        groups = np.unique(prot_vals)

        group_mus: Dict[Any, np.ndarray] = {}
        group_sizes: Dict[Any, int] = {}

        for g in groups:
            mask = prot_vals == g
            X_g = X[mask]
            n_g = int(mask.sum())
            group_sizes[g] = n_g

            if n_g == 0:
                continue

            # Select background for this group
            bg = q_a_dict.get(g, q_a_dict.get("__global__", q_pool))

            local_seed = int(rng.integers(0, 2**31))
            shap_g = compute_shap(
                model=model, x_eval=X_g, background=bg,
                n_coalitions=self.n_coalitions, seed=local_seed,
            )

            # Mean SHAP vector for this group
            mu_g = shap_g.mean(axis=0)

            # DP: clip + noise
            mu_g_clipped = clip_l2(mu_g, C=C)
            if sigma_dp > 0:
                from ..privacy.dp_noise import add_gaussian_noise
                noise_seed = int(rng.integers(0, 2**31))
                mu_g_clipped = add_gaussian_noise(
                    mu_g_clipped, sigma=sigma_dp * C, seed=noise_seed
                )

            group_mus[g] = mu_g_clipped

        return {
            "client_id": client_data.get("client_id", -1),
            "group_mus": group_mus,
            "group_sizes": group_sizes,
            "n": n,
        }

    # -----------------------------------------------------------------------
    # Algorithm 2: Server-side bias-aware aggregation
    # -----------------------------------------------------------------------

    def server_aggregate(
        self,
        client_summaries: List[Dict],
        z_tau: float = 3.5,
        s_min: float = 0.001,
        beta_trim: float = 0.1,
    ) -> Dict:
        """Algorithm 2: Server-side bias-aware aggregation with drift detection.

        Steps:
          1. Collect per-client per-group mean SHAP vectors.
          2. Compute Z-scores for drift detection; flag outlier clients.
          3. Weighted (by group size) aggregation of non-outlier clients.
          4. Compute global mean and group-level ADI_norm.

        Args:
            client_summaries: List of dicts from client_compute().
            z_tau: Z-score threshold for outlier detection.
            s_min: Minimum std floor for ADI computation.
            beta_trim: Fraction for trimmed mean (fallback).

        Returns:
            Dict with:
              'global_mu': global mean SHAP vector,
              'group_mus': {group: weighted mean SHAP},
              'adi_norm': scalar ADI_norm,
              'outlier_clients': list of client_ids flagged as outliers,
              'n_clients_used': number of non-outlier clients,
        """
        from ..metrics.adi import compute_adi_norm

        # Collect all group-level mean vectors across clients
        # Structure: group_val -> list of (mu, size)
        group_collection: Dict[Any, List[Tuple[np.ndarray, int]]] = {}
        client_global_mus: List[np.ndarray] = []
        client_ns: List[float] = []

        for cs in client_summaries:
            if cs["n"] == 0:
                continue
            for g, mu in cs["group_mus"].items():
                if g not in group_collection:
                    group_collection[g] = []
                group_collection[g].append((mu, cs["group_sizes"].get(g, 1)))
            # Global mu for this client = weighted avg of group mus
            if cs["group_mus"]:
                all_mus = np.array(list(cs["group_mus"].values()))
                all_sizes = np.array([cs["group_sizes"].get(g, 1) for g in cs["group_mus"]])
                weights = all_sizes / all_sizes.sum()
                client_mu = (all_mus * weights[:, None]).sum(axis=0)
                client_global_mus.append(client_mu)
                client_ns.append(float(cs["n"]))

        if not client_global_mus:
            raise ValueError("No valid client summaries to aggregate.")

        # Drift detection via Z-scores on L2 norms
        client_global_mus_arr = np.array(client_global_mus)
        norms = np.linalg.norm(client_global_mus_arr, axis=1)
        z_scores = self.drift_z_scores(client_global_mus, None)
        outlier_mask = z_scores > z_tau
        n_outliers = int(outlier_mask.sum())

        if n_outliers > 0:
            logger.info("Detected %d outlier clients (z_tau=%.2f).", n_outliers, z_tau)

        # Use non-outlier clients
        valid_idx = np.where(~outlier_mask)[0]
        if len(valid_idx) == 0:
            logger.warning("All clients flagged as outliers; using all.")
            valid_idx = np.arange(len(client_global_mus))

        valid_mus = client_global_mus_arr[valid_idx]
        valid_ns = np.array(client_ns)[valid_idx]
        weights = valid_ns / valid_ns.sum()
        global_mu = (valid_mus * weights[:, None]).sum(axis=0)

        # Per-group aggregation
        group_mus: Dict[Any, np.ndarray] = {}
        for g, entries in group_collection.items():
            mus_g = np.array([e[0] for e in entries])
            sizes_g = np.array([float(e[1]) for e in entries])
            w_g = sizes_g / sizes_g.sum()
            group_mus[g] = (mus_g * w_g[:, None]).sum(axis=0)

        # ADI_norm
        all_groups = [g for g in group_mus if g != "__all__" and g != "__global__"]
        if len(all_groups) >= 2:
            prot_labels = []
            mus_for_adi = []
            for i, g in enumerate(all_groups):
                prot_labels.append(i)
                mus_for_adi.append(group_mus[g])
            adi = compute_adi_norm(mus_for_adi, np.array(prot_labels), s_min=s_min)
        else:
            adi = 0.0

        outlier_client_ids = [
            client_summaries[i]["client_id"]
            for i in np.where(outlier_mask)[0]
            if i < len(client_summaries)
        ]

        return {
            "global_mu": global_mu,
            "group_mus": group_mus,
            "adi_norm": float(adi),
            "outlier_clients": outlier_client_ids,
            "n_clients_used": int(len(valid_idx)),
            "z_scores": z_scores.tolist(),
        }

    # -----------------------------------------------------------------------
    # Drift Z-scores
    # -----------------------------------------------------------------------

    def drift_z_scores(
        self,
        client_mus: List[np.ndarray],
        global_mu: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute per-client drift Z-scores based on L2 norm of mean SHAP vectors.

        Z_k = (||mu_k||_2 - mean_norm) / std_norm

        Args:
            client_mus: Per-client mean SHAP vectors.
            global_mu: Global mean (unused in norm-based Z-score; kept for API compat).

        Returns:
            Z-score array, shape (K,).
        """
        norms = np.array([np.linalg.norm(mu) for mu in client_mus])
        mean_norm = norms.mean()
        std_norm = norms.std()
        if std_norm < 1e-12:
            return np.zeros_like(norms)
        return np.abs((norms - mean_norm) / std_norm)

    # -----------------------------------------------------------------------
    # Trimmed mean
    # -----------------------------------------------------------------------

    def trimmed_mean(
        self,
        vectors: List[np.ndarray],
        beta: float = 0.1,
    ) -> np.ndarray:
        """Compute beta-trimmed mean of a list of vectors.

        Removes the bottom and top (beta * 100)% by L2 norm and averages the rest.

        Args:
            vectors: List of 1D arrays (same length d each).
            beta: Trimming fraction (0 <= beta < 0.5).

        Returns:
            Trimmed mean vector, shape (d,).
        """
        arr = np.array(vectors)  # (K, d)
        K = len(arr)
        n_trim = int(K * beta)

        norms = np.linalg.norm(arr, axis=1)
        sorted_idx = np.argsort(norms)
        kept_idx = sorted_idx[n_trim: K - n_trim] if n_trim > 0 else sorted_idx
        return arr[kept_idx].mean(axis=0)

    # -----------------------------------------------------------------------
    # Full protocol
    # -----------------------------------------------------------------------

    def run_full_protocol(
        self,
        client_datasets: List[Dict],
        global_model: Any,
        ref_data: np.ndarray,
        ref_protected: np.ndarray,
        config: Dict,
    ) -> Dict:
        """Run the complete BA-FedSHAP protocol.

        Args:
            client_datasets: List of per-client data dicts.
            global_model: Trained global FL model.
            ref_data: Reference data for background construction (e.g. validation set).
            ref_protected: Protected attribute values for ref_data.
            config: Configuration dict with keys:
                - background_size: K_global
                - dp_clip_c: C
                - dp_sigma: sigma for DP (0 = no DP)
                - drift_z_tau: Z-score threshold
                - n_min_group: Minimum group size threshold
                - kernelshap_coalitions: M

        Returns:
            Dict with server aggregate output + per-client summaries.
        """
        K_global = config.get("background_size", self.K_global)
        C = config.get("dp_clip_c", 1.0)
        sigma_dp = config.get("dp_sigma", 0.0)
        z_tau = config.get("drift_z_tau", 3.5)
        n_coalitions = config.get("kernelshap_coalitions", self.n_coalitions)
        seed = config.get("seed", self.seed)

        self.n_coalitions = n_coalitions

        logger.info(
            "BA-FedSHAP: K=%d, C=%.2f, sigma=%.4f, z_tau=%.2f, M=%d",
            K_global, C, sigma_dp, z_tau, n_coalitions,
        )

        # Step 1: Construct stratified backgrounds (Algorithm 3)
        backgrounds = self.construct_backgrounds(
            ref_data=ref_data,
            protected_col=ref_protected,
            K_global=K_global,
            seed=seed,
        )
        q_pool = backgrounds["__global__"]

        # Step 2: Client computation (Algorithm 1)
        client_summaries = []
        for k, ds in enumerate(client_datasets):
            if ds["n"] == 0:
                continue
            cs = self.client_compute(
                client_data=ds,
                model=global_model,
                q_pool=q_pool,
                q_a_dict=backgrounds,
                C=C,
                sigma_dp=sigma_dp,
                seed=seed + k,
            )
            client_summaries.append(cs)
            logger.debug("Client %d done (n=%d).", k, ds["n"])

        # Step 3: Server aggregation (Algorithm 2)
        result = self.server_aggregate(
            client_summaries=client_summaries,
            z_tau=z_tau,
        )
        result["client_summaries"] = client_summaries
        result["config"] = config

        return result
