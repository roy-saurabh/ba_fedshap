#!/usr/bin/env python3
# LEGACY — reads results.json (old format). The current pipeline writes full_eval.json.
# Use make_paper_tables.py (all tables) or make_softx_compas_table.py (Table 1) instead.
"""Aggregate raw results into paper tables.

Reads all results/raw/{dataset}/seed_*/alpha_*/results.json files,
aggregates over seeds, and writes:
    results/processed/{dataset}_aggregated.csv
    results/tables/table_iv_primary_reliability.{csv,md}
    results/tables/table_v_faithfulness_sanity.{csv,md}
    results/tables/table_vi_ablation.{csv,md}
    results/tables/table_vii_dp_noise.{csv,md}
    results/tables/table_viii_adi_outcome_correlation.{csv,md}
    results/tables/table_ix_lipschitz.{csv,md}
    results/tables/table_x_runtime.{csv,md}
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = REPO_ROOT / "results" / "raw"
PROC_DIR = REPO_ROOT / "results" / "processed"
TABLE_DIR = REPO_ROOT / "results" / "tables"


# ---------------------------------------------------------------------------
# Load all result files
# ---------------------------------------------------------------------------

def load_all_results() -> List[Dict]:
    records = []
    for result_file in sorted(RAW_DIR.rglob("results.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
            data["_file"] = str(result_file)
            records.append(data)
        except Exception as exc:
            logger.warning("Could not read %s: %s", result_file, exc)
    logger.info("Loaded %d result files.", len(records))
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_std(vals: List[float]) -> str:
    if not vals:
        return "—"
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to Markdown table string."""
    lines = []
    lines.append("| " + " | ".join(str(c) for c in df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, stem: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df_to_markdown(df))
    logger.info("Saved %s", csv_path)
    logger.info("Saved %s", md_path)


# ---------------------------------------------------------------------------
# Table IV: Primary reliability (ADI_norm per method, per dataset)
# ---------------------------------------------------------------------------

def make_table_iv(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        dataset = r.get("dataset", "unknown")
        seed = r.get("seed", 0)
        alpha = r.get("alpha", 0.5)

        # BA-FedSHAP at inf (no DP)
        baf = r.get("ba_fedshap", {})
        adi_inf = baf.get("inf", {}).get("adi_norm", float("nan"))

        # Baselines
        baselines = r.get("baselines", {})
        for method, bdata in baselines.items():
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "alpha": alpha,
                "method": method,
                "adi_norm": bdata.get("adi_norm", float("nan")),
            })
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "alpha": alpha,
            "method": "ba_fedshap_inf",
            "adi_norm": adi_inf,
        })

    if not rows:
        return pd.DataFrame(columns=["dataset", "method", "adi_norm_mean", "adi_norm_std"])

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["dataset", "method"])["adi_norm"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "adi_norm_mean", "std": "adi_norm_std"})
    )
    return summary


# ---------------------------------------------------------------------------
# Table V: Faithfulness and sanity (placeholder)
# ---------------------------------------------------------------------------

def make_table_v(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        dataset = r.get("dataset", "unknown")
        for method, bdata in r.get("baselines", {}).items():
            rows.append({
                "dataset": dataset,
                "method": method,
                "deletion_auc": bdata.get("deletion_auc", float("nan")),
                "insertion_auc": bdata.get("insertion_auc", float("nan")),
                "param_rand_cosine": bdata.get("param_rand_cosine", float("nan")),
                "label_rand_spearman": bdata.get("label_rand_spearman", float("nan")),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "dataset", "method", "deletion_auc", "insertion_auc",
            "param_rand_cosine", "label_rand_spearman",
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table VI: Ablation study
# ---------------------------------------------------------------------------

def make_table_vi(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        dataset = r.get("dataset", "unknown")
        if dataset != "adult":
            continue
        ablation = r.get("ablation", {})
        for variant, vdata in ablation.items():
            rows.append({
                "variant": variant,
                "adi_norm": vdata.get("adi_norm", float("nan")),
                "deletion_auc": vdata.get("deletion_auc", float("nan")),
                "runtime_s": vdata.get("runtime_s", float("nan")),
            })
    if not rows:
        return pd.DataFrame(columns=["variant", "adi_norm", "deletion_auc", "runtime_s"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table VII: DP noise sweep
# ---------------------------------------------------------------------------

def make_table_vii(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        dataset = r.get("dataset", "unknown")
        for eps_label, edata in r.get("ba_fedshap", {}).items():
            if isinstance(edata, dict) and "error" not in edata:
                rows.append({
                    "dataset": dataset,
                    "epsilon": eps_label,
                    "seed": r.get("seed"),
                    "adi_norm": edata.get("adi_norm", float("nan")),
                    "runtime_s": edata.get("runtime_s", float("nan")),
                })
    if not rows:
        return pd.DataFrame(columns=["dataset", "epsilon", "adi_norm_mean", "adi_norm_std"])
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["dataset", "epsilon"])["adi_norm"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "adi_norm_mean", "std": "adi_norm_std"})
    )
    return summary


# ---------------------------------------------------------------------------
# Table VIII: ADI–outcome correlation
# ---------------------------------------------------------------------------

def make_table_viii(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        corr = r.get("adi_outcome_correlation", {})
        if corr:
            rows.append({
                "dataset": r.get("dataset"),
                "method": corr.get("method"),
                "spearman_rho": corr.get("spearman_rho", float("nan")),
                "p_value": corr.get("p_value", float("nan")),
            })
    if not rows:
        return pd.DataFrame(columns=["dataset", "method", "spearman_rho", "p_value"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table IX: Lipschitz constants
# ---------------------------------------------------------------------------

def make_table_ix(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for method, bdata in r.get("baselines", {}).items():
            lip = bdata.get("lipschitz", {})
            if lip:
                rows.append({
                    "dataset": r.get("dataset"),
                    "method": method,
                    "l_hat": lip.get("l_hat", float("nan")),
                    "l_hat_mean": lip.get("l_hat_mean", float("nan")),
                })
        baf = r.get("ba_fedshap", {}).get("inf", {})
        if "lipschitz" in baf:
            rows.append({
                "dataset": r.get("dataset"),
                "method": "ba_fedshap_inf",
                "l_hat": baf["lipschitz"].get("l_hat", float("nan")),
                "l_hat_mean": baf["lipschitz"].get("l_hat_mean", float("nan")),
            })
    if not rows:
        return pd.DataFrame(columns=["dataset", "method", "l_hat", "l_hat_mean"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table X: Runtime comparison
# ---------------------------------------------------------------------------

def make_table_x(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        dataset = r.get("dataset", "unknown")
        seed = r.get("seed")

        for eps_label, edata in r.get("ba_fedshap", {}).items():
            if isinstance(edata, dict) and "runtime_s" in edata:
                rows.append({
                    "dataset": dataset,
                    "method": f"ba_fedshap_eps{eps_label}",
                    "seed": seed,
                    "runtime_s": edata["runtime_s"],
                })

        for method, bdata in r.get("baselines", {}).items():
            if isinstance(bdata, dict) and "runtime_s" in bdata:
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "runtime_s": bdata["runtime_s"],
                })

    if not rows:
        return pd.DataFrame(columns=["dataset", "method", "runtime_s_mean"])
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["dataset", "method"])["runtime_s"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "runtime_s_mean", "std": "runtime_s_std"})
    )
    return summary


# ---------------------------------------------------------------------------
# Processed aggregated CSV
# ---------------------------------------------------------------------------

def make_aggregated_csv(records: List[Dict]) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    by_dataset: Dict[str, List[Dict]] = {}
    for r in records:
        ds = r.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    for ds, ds_records in by_dataset.items():
        rows = []
        for r in ds_records:
            row = {
                "seed": r.get("seed"),
                "alpha": r.get("alpha"),
                "eval_accuracy": r.get("eval_accuracy"),
            }
            for eps_label, edata in r.get("ba_fedshap", {}).items():
                if isinstance(edata, dict):
                    row[f"ba_fedshap_eps{eps_label}_adi"] = edata.get("adi_norm")
            rows.append(row)

        df = pd.DataFrame(rows)
        out = PROC_DIR / f"{ds}_aggregated.csv"
        df.to_csv(out, index=False)
        logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    records = load_all_results()

    if not records:
        logger.warning("No result files found in %s. Tables will be empty.", RAW_DIR)

    make_aggregated_csv(records)

    # Table IV
    df4 = make_table_iv(records)
    save_table(df4, "table_iv_primary_reliability")

    # Table V
    df5 = make_table_v(records)
    save_table(df5, "table_v_faithfulness_sanity")

    # Table VI
    df6 = make_table_vi(records)
    save_table(df6, "table_vi_ablation")

    # Table VII
    df7 = make_table_vii(records)
    save_table(df7, "table_vii_dp_noise")

    # Table VIII
    df8 = make_table_viii(records)
    save_table(df8, "table_viii_adi_outcome_correlation")

    # Table IX
    df9 = make_table_ix(records)
    save_table(df9, "table_ix_lipschitz")

    # Table X
    df10 = make_table_x(records)
    save_table(df10, "table_x_runtime")

    logger.info("make_tables.py complete.")


if __name__ == "__main__":
    main()
