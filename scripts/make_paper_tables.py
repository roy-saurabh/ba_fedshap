#!/usr/bin/env python3
"""Generate all paper tables (IV–X) from full_eval.json result files.

Reads:  results/raw/{dataset}/seed_{seed}/alpha_{alpha}/full_eval.json
Writes: results/tables/table_{iv,v,vi,vii,viii,ix,x}.{csv,md}
        results/processed/aggregated_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = REPO_ROOT / "results" / "raw"
TABLE_DIR = REPO_ROOT / "results" / "tables"
PROC_DIR = REPO_ROOT / "results" / "processed"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load all result files
# ---------------------------------------------------------------------------

def load_all() -> list[dict]:
    records = []
    for f in sorted(RAW_DIR.rglob("full_eval.json")):
        try:
            d = json.loads(f.read_text())
            d["_file"] = str(f)
            records.append(d)
        except Exception as e:
            logger.warning("Skipping %s: %s", f, e)
    logger.info("Loaded %d full_eval.json files", len(records))
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(vals: list[float]) -> str:
    if not vals:
        return "—"
    m, s = np.nanmean(vals), np.nanstd(vals)
    return f"{m:.3f} ± {s:.3f}"


def _ci_str(vals: list[float]) -> str:
    if not vals:
        return "—"
    lo, hi = np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)
    return f"[{lo:.3f},{hi:.3f}]"


def save_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    # Markdown
    md = "| " + " | ".join(str(c) for c in df.columns) + " |\n"
    md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for _, row in df.iterrows():
        md += "| " + " | ".join(str(v) for v in row.values) + " |\n"
    (TABLE_DIR / f"{stem}.md").write_text(md)
    logger.info("Saved %s", TABLE_DIR / f"{stem}.csv")


# ---------------------------------------------------------------------------
# Table IV — Primary Reliability (oracle estimation error)
# Filter to alpha=0.5
# ---------------------------------------------------------------------------

def make_table_iv(records: list[dict]) -> pd.DataFrame:
    """Oracle estimation error, L1, Spearman rho, CI coverage.
    Averaged over seeds × datasets at Dir(alpha=0.5).
    """
    methods = [
        ("centralized_oracle", "Centralized Oracle SHAP"),
        ("local_shap", "Local SHAP"),
        ("naive_aggregated", "Naive Aggregated SHAP"),
        ("shared_background", "Shared-background SHAP"),
        ("kmeans_background", "k-means background SHAP"),
        ("ba_fedshap_inf", "BA-FedSHAP (ε=∞)"),
        ("ba_fedshap_4", "BA-FedSHAP (ε=4)"),
        ("ba_fedshap_1", "BA-FedSHAP (ε=1)"),
    ]

    stats_by_method: dict[str, dict] = {k: {"l1": [], "rho": [], "adi_err": [], "ci_cov": []}
                                         for k, _ in methods}

    for r in records:
        if abs(r.get("alpha", 0) - 0.5) > 0.01:
            continue

        bl = r.get("baselines", {})
        baf = r.get("ba_fedshap", {})

        for key, _ in methods:
            if key.startswith("ba_fedshap_"):
                eps = key.split("_")[-1]
                v = baf.get(eps, {})
            else:
                v = bl.get(key, {})

            if "error" in v or not v:
                continue

            stats_by_method[key]["l1"].append(v.get("l1_to_global_mean", float("nan")))
            stats_by_method[key]["rho"].append(v.get("spearman_rho_mean", float("nan")))
            stats_by_method[key]["adi_err"].append(v.get("adi_error", float("nan")))
            if key.startswith("ba_fedshap_"):
                stats_by_method[key]["ci_cov"].append(v.get("ci_covers_oracle", float("nan")))

    rows = []
    for key, label in methods:
        s = stats_by_method[key]
        l1_vals = [x for x in s["l1"] if not np.isnan(x)]
        rho_vals = [x for x in s["rho"] if not np.isnan(x)]
        err_vals = [x for x in s["adi_err"] if not np.isnan(x)]
        cov_vals = [x for x in s["ci_cov"] if not np.isnan(x)]

        row = {
            "Method": label,
            "L1 to oracle ↓": _ms(l1_vals),
            "Spearman ρ ↑": _ms(rho_vals),
            "|ADI − oracle| ↓": _ms(err_vals),
            "CI coverage": f"{np.nanmean(cov_vals):.2f}" if cov_vals else "—",
            "n_obs": len(l1_vals),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table V — Faithfulness and Sanity
# ---------------------------------------------------------------------------

def make_table_v(records: list[dict]) -> pd.DataFrame:
    methods = [
        ("local_shap", "Local SHAP"),
        ("shared_background", "Shared-background SHAP"),
        ("kmeans_background", "k-means background SHAP"),
        ("ba_fedshap_inf", "BA-FedSHAP (Q_pool, ε=∞)"),
    ]

    stats_by: dict[str, dict] = {k: {"del": [], "ins": []} for k, _ in methods}

    for r in records:
        if abs(r.get("alpha", 0) - 0.5) > 0.01:
            continue
        bl = r.get("baselines", {})
        baf = r.get("ba_fedshap", {})

        for key, _ in methods:
            if key == "ba_fedshap_inf":
                v = baf.get("inf", {})
            else:
                v = bl.get(key, {})
            if "error" in v or not v:
                continue
            stats_by[key]["del"].append(v.get("deletion_auc", float("nan")))
            stats_by[key]["ins"].append(v.get("insertion_auc", float("nan")))

    rows = []
    for key, label in methods:
        s = stats_by[key]
        del_vals = [x for x in s["del"] if not np.isnan(x)]
        ins_vals = [x for x in s["ins"] if not np.isnan(x)]
        rows.append({
            "Method": label,
            "Deletion AUC ↓": _ms(del_vals),
            "Insertion AUC ↑": _ms(ins_vals),
            "n_obs": len(del_vals),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table VI — Ablation (from run_ablation.py results)
# ---------------------------------------------------------------------------

def make_table_vi(records_ablation: list[dict]) -> pd.DataFrame:
    variant_labels = {
        "full": "(A) Full BA-FedSHAP (ε=∞)",
        "no_drift": "(D) − drift Z-test",
        "no_stratified_bg": "(B) − group conditioning",
        "with_dp_eps4": "(H) + DP ε=4",
        "with_dp_eps1": "(I) + DP ε=1",
    }
    stats_by: dict[str, dict] = {k: {"adi": [], "del": []} for k in variant_labels}

    for r in records_ablation:
        abl = r.get("ablation", {})
        for variant, v in abl.items():
            if variant not in stats_by or "error" in v:
                continue
            stats_by[variant]["adi"].append(v.get("adi_norm", float("nan")))
            stats_by[variant]["del"].append(v.get("deletion_auc", float("nan")))

    rows = []
    for key, label in variant_labels.items():
        s = stats_by[key]
        adi_vals = [x for x in s["adi"] if not np.isnan(x)]
        del_vals = [x for x in s["del"] if not np.isnan(x)]
        rows.append({
            "Configuration": label,
            "ADI_norm": _ms(adi_vals),
            "Deletion AUC ↓": _ms(del_vals),
            "n_obs": len(adi_vals),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table VII — DP Noise Sensitivity
# ---------------------------------------------------------------------------

def make_table_vii(records: list[dict]) -> pd.DataFrame:
    epsilons = ["inf", "8", "4", "2", "1"]
    stats_by: dict[str, dict] = {e: {"adi_err": [], "rho": [], "ci_cov": []}
                                   for e in epsilons}

    for r in records:
        if abs(r.get("alpha", 0) - 0.5) > 0.01:
            continue
        baf = r.get("ba_fedshap", {})
        for eps in epsilons:
            v = baf.get(eps, {})
            if "error" in v or not v:
                continue
            stats_by[eps]["adi_err"].append(v.get("adi_error", float("nan")))
            stats_by[eps]["rho"].append(v.get("spearman_rho_mean", float("nan")))
            stats_by[eps]["ci_cov"].append(v.get("ci_covers_oracle", float("nan")))

    rows = []
    for eps in epsilons:
        s = stats_by[eps]
        err_vals = [x for x in s["adi_err"] if not np.isnan(x)]
        rho_vals = [x for x in s["rho"] if not np.isnan(x)]
        cov_vals = [x for x in s["ci_cov"] if not np.isnan(x)]
        rows.append({
            "ε": "∞" if eps == "inf" else eps,
            "|ADI − oracle| ↓": _ms(err_vals),
            "Spearman ρ ↑": _ms(rho_vals),
            "CI covers oracle": f"{np.nanmean(cov_vals):.2f}" if cov_vals else "—",
            "n_obs": len(err_vals),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table VIII — ADI–Outcome Correlation
# (uses ADI_norm vs model accuracy as proxy for fairness metric)
# ---------------------------------------------------------------------------

def make_table_viii(records: list[dict]) -> pd.DataFrame:
    datasets = ["adult", "compas", "german", "bank"]
    rows = []

    for ds in datasets:
        ds_records = [r for r in records if r.get("dataset") == ds]
        if not ds_records:
            continue

        adi_vals = []
        acc_vals = []
        for r in ds_records:
            baf_inf = r.get("ba_fedshap", {}).get("inf", {})
            if "error" not in baf_inf and baf_inf:
                adi_vals.append(baf_inf.get("adi_norm", float("nan")))
                acc_vals.append(r.get("eval_accuracy", float("nan")))

        adi_vals = [x for x in adi_vals if not np.isnan(x)]
        acc_vals = [x for x in acc_vals if not np.isnan(x)]

        if len(adi_vals) >= 3:
            rho, pval = stats.pearsonr(adi_vals, acc_vals)
            n = len(adi_vals)
            # Fisher z CI
            z = np.arctanh(rho)
            se = 1.0 / np.sqrt(max(n - 3, 1))
            ci_lo = float(np.tanh(z - 1.96 * se))
            ci_hi = float(np.tanh(z + 1.96 * se))
            pval_str = "<0.001" if pval < 0.001 else f"{pval:.3f}"
            rows.append({
                "Dataset": ds.capitalize(),
                "r(ADI, accuracy)": f"{rho:.2f}",
                "95% CI": f"[{ci_lo:.2f},{ci_hi:.2f}]",
                "p": pval_str,
                "n": n,
            })
        else:
            rows.append({"Dataset": ds.capitalize(), "r(ADI, accuracy)": "—",
                         "95% CI": "—", "p": "—", "n": len(adi_vals)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table IX — Lipschitz Estimates
# ---------------------------------------------------------------------------

def make_table_ix(records: list[dict]) -> pd.DataFrame:
    datasets = ["adult", "compas", "german", "bank"]
    rows = []

    for ds in datasets:
        ds_records = [r for r in records if r.get("dataset") == ds
                      and abs(r.get("alpha", 0) - 0.5) < 0.01]
        if not ds_records:
            continue

        l_hats = [r["lipschitz"]["l_hat_mean"] for r in ds_records
                  if "l_hat_mean" in r.get("lipschitz", {})]
        if l_hats:
            rows.append({
                "Dataset / Model": f"{ds.capitalize()} / LogReg",
                "d": ds_records[0].get("n_features", "—"),
                "L̂ mean": f"{np.nanmean(l_hats):.2f} ± {np.nanstd(l_hats):.2f}",
                "n_obs": len(l_hats),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table X — Runtime
# ---------------------------------------------------------------------------

def make_table_x(records: list[dict]) -> pd.DataFrame:
    methods = [
        ("ba_fedshap_inf", "BA-FedSHAP (ε=∞)"),
        ("shared_background", "Shared-background SHAP"),
        ("local_shap", "Local SHAP"),
    ]
    # Only adult, alpha=0.5
    adult_records = [r for r in records
                     if r.get("dataset") == "adult" and abs(r.get("alpha", 0) - 0.5) < 0.01]

    rows = []
    for key, label in methods:
        rts = []
        for r in adult_records:
            if key == "ba_fedshap_inf":
                v = r.get("ba_fedshap", {}).get("inf", {})
            else:
                v = r.get("baselines", {}).get(key, {})
            if v and "runtime_s" in v:
                rts.append(v["runtime_s"])
        rows.append({
            "Stage / Method": label,
            "Runtime (s)": _ms(rts),
            "n_obs": len(rts),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregated summary JSON for manuscript
# ---------------------------------------------------------------------------

def make_summary(records: list[dict]) -> dict:
    """Build a summary dict with all key metrics for manuscript filling."""
    summary = {}

    # Table IV: primary metrics at alpha=0.5, all 3 main datasets
    alpha05 = [r for r in records if abs(r.get("alpha", 0) - 0.5) < 0.01
               and r.get("dataset") in ("adult", "compas", "german")]

    methods_iv = {
        "local_shap": ("baselines", "local_shap"),
        "naive_aggregated": ("baselines", "naive_aggregated"),
        "shared_background": ("baselines", "shared_background"),
        "kmeans_background": ("baselines", "kmeans_background"),
        "ba_fedshap_inf": ("ba_fedshap", "inf"),
        "ba_fedshap_4": ("ba_fedshap", "4"),
        "ba_fedshap_1": ("ba_fedshap", "1"),
    }

    table_iv = {}
    for mkey, (src, eps_or_key) in methods_iv.items():
        l1s, rhos, errs, covs = [], [], [], []
        for r in alpha05:
            grp = r.get(src, {})
            v = grp.get(eps_or_key, {})
            if v and "error" not in v:
                l1s.append(v.get("l1_to_global_mean", float("nan")))
                rhos.append(v.get("spearman_rho_mean", float("nan")))
                errs.append(v.get("adi_error", float("nan")))
                if src == "ba_fedshap":
                    covs.append(v.get("ci_covers_oracle", float("nan")))

        table_iv[mkey] = {
            "l1_mean": float(np.nanmean(l1s)) if l1s else None,
            "l1_std": float(np.nanstd(l1s)) if l1s else None,
            "rho_mean": float(np.nanmean(rhos)) if rhos else None,
            "rho_std": float(np.nanstd(rhos)) if rhos else None,
            "adi_err_mean": float(np.nanmean(errs)) if errs else None,
            "adi_err_std": float(np.nanstd(errs)) if errs else None,
            "ci_lo": float(np.nanpercentile(errs, 2.5)) if errs else None,
            "ci_hi": float(np.nanpercentile(errs, 97.5)) if errs else None,
            "ci_coverage": float(np.nanmean(covs)) if covs else None,
            "n": len(l1s),
        }

    summary["table_iv"] = table_iv

    # Table VII: DP sweep
    dp_recs = [r for r in records if r.get("dataset") == "adult"
               and abs(r.get("alpha", 0) - 0.5) < 0.01]
    table_vii = {}
    for eps in ["inf", "8", "4", "2", "1"]:
        errs = [r.get("ba_fedshap", {}).get(eps, {}).get("adi_error", float("nan"))
                for r in dp_recs if "error" not in r.get("ba_fedshap", {}).get(eps, {})]
        errs = [x for x in errs if not np.isnan(x)]
        rhos = [r.get("ba_fedshap", {}).get(eps, {}).get("spearman_rho_mean", float("nan"))
                for r in dp_recs if "error" not in r.get("ba_fedshap", {}).get(eps, {})]
        rhos = [x for x in rhos if not np.isnan(x)]
        covs = [r.get("ba_fedshap", {}).get(eps, {}).get("ci_covers_oracle", float("nan"))
                for r in dp_recs if "error" not in r.get("ba_fedshap", {}).get(eps, {})]
        covs = [x for x in covs if not np.isnan(x)]
        table_vii[eps] = {
            "adi_err_mean": float(np.nanmean(errs)) if errs else None,
            "adi_err_std": float(np.nanstd(errs)) if errs else None,
            "rho_mean": float(np.nanmean(rhos)) if rhos else None,
            "ci_coverage": float(np.nanmean(covs)) if covs else None,
            "n": len(errs),
        }
    summary["table_vii"] = table_vii

    # Table V: faithfulness
    faith_recs = [r for r in records if abs(r.get("alpha", 0) - 0.5) < 0.01]
    table_v = {}
    for mkey, src_key in [("local_shap", ("baselines", "local_shap")),
                           ("shared_background", ("baselines", "shared_background")),
                           ("kmeans_background", ("baselines", "kmeans_background")),
                           ("ba_fedshap_inf", ("ba_fedshap", "inf"))]:
        dels, ins = [], []
        for r in faith_recs:
            v = r.get(src_key[0], {}).get(src_key[1], {})
            if v and "error" not in v:
                dels.append(v.get("deletion_auc", float("nan")))
                ins.append(v.get("insertion_auc", float("nan")))
        dels = [x for x in dels if not np.isnan(x)]
        ins = [x for x in ins if not np.isnan(x)]
        table_v[mkey] = {
            "del_mean": float(np.nanmean(dels)) if dels else None,
            "del_std": float(np.nanstd(dels)) if dels else None,
            "ins_mean": float(np.nanmean(ins)) if ins else None,
            "ins_std": float(np.nanstd(ins)) if ins else None,
            "n": len(dels),
        }
    summary["table_v"] = table_v

    # Lipschitz
    lip_vals = {}
    for ds in ["adult", "compas", "german", "bank"]:
        ds_r = [r for r in records if r.get("dataset") == ds
                and abs(r.get("alpha", 0) - 0.5) < 0.01
                and "l_hat_mean" in r.get("lipschitz", {})]
        if ds_r:
            vals = [r["lipschitz"]["l_hat_mean"] for r in ds_r]
            lip_vals[ds] = {
                "l_hat_mean": float(np.nanmean(vals)),
                "l_hat_std": float(np.nanstd(vals)),
                "n_features": ds_r[0].get("n_features"),
            }
    summary["lipschitz"] = lip_vals

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    records = load_all()

    if not records:
        logger.error("No full_eval.json files found.")
        sys.exit(1)

    # Load ablation results
    ablation_records = []
    for f in sorted(RAW_DIR.rglob("ablation/ablation_*.json")):
        try:
            ablation_records.append(json.loads(f.read_text()))
        except Exception as e:
            logger.warning("Skipping %s: %s", f, e)
    logger.info("Loaded %d ablation files", len(ablation_records))

    # Generate tables
    df_iv = make_table_iv(records)
    save_table(df_iv, "table_iv_primary_reliability")

    df_v = make_table_v(records)
    save_table(df_v, "table_v_faithfulness")

    df_vi = make_table_vi(ablation_records)
    save_table(df_vi, "table_vi_ablation")

    df_vii = make_table_vii(records)
    save_table(df_vii, "table_vii_dp_noise")

    df_viii = make_table_viii(records)
    save_table(df_viii, "table_viii_adi_outcome_correlation")

    df_ix = make_table_ix(records)
    save_table(df_ix, "table_ix_lipschitz")

    df_x = make_table_x(records)
    save_table(df_x, "table_x_runtime")

    # Summary JSON for manuscript filling
    summary = make_summary(records)
    out_sum = PROC_DIR / "manuscript_values.json"
    out_sum.write_text(json.dumps(summary, indent=2))
    logger.info("Summary saved: %s", out_sum)

    # Print Table IV to console
    print("\n=== TABLE IV: Primary Reliability ===")
    print(df_iv.to_string(index=False))
    print("\n=== TABLE V: Faithfulness ===")
    print(df_v.to_string(index=False))
    print("\n=== TABLE VII: DP Noise Sensitivity ===")
    print(df_vii.to_string(index=False))
    print("\n=== TABLE VIII: ADI-Outcome Correlation ===")
    print(df_viii.to_string(index=False))
    print("\n=== TABLE IX: Lipschitz Estimates ===")
    print(df_ix.to_string(index=False))
    print("\n=== TABLE X: Runtime ===")
    print(df_x.to_string(index=False))

    logger.info("make_paper_tables.py complete.")


if __name__ == "__main__":
    main()
