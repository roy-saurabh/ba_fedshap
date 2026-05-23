#!/usr/bin/env python3
"""Generate SoftwareX Table 1 from the 15-cell COMPAS full_eval.json corpus.

Reads:  <input>/seed_<seed>/alpha_<alpha>/full_eval.json  (15 files)
Writes: <output>  (CSV)  — aggregate means over 15 cells

Usage
-----
python scripts/make_softx_compas_table.py \
    --input results/raw/compas \
    --output results/tables/table1_compas_aggregate.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METHODS = [
    ("local_shap",        "baselines", "local_shap",        "Local SHAP"),
    ("naive_aggregated",  "baselines", "naive_aggregated",   "Naive aggregated SHAP"),
    ("shared_background", "baselines", "shared_background",  "Shared-background SHAP"),
    ("kmeans_background", "baselines", "kmeans_background",  "k-means background SHAP"),
    ("ba_inf",            "ba_fedshap", "inf",               "BA-FedSHAP ε=∞"),
    ("ba_8",              "ba_fedshap", "8",                 "BA-FedSHAP ε=8"),
    ("ba_4",              "ba_fedshap", "4",                 "BA-FedSHAP ε=4"),
    ("ba_2",              "ba_fedshap", "2",                 "BA-FedSHAP ε=2"),
    ("ba_1",              "ba_fedshap", "1",                 "BA-FedSHAP ε=1"),
]


def load_cells(input_dir: Path) -> list[dict]:
    records = []
    for f in sorted(input_dir.rglob("full_eval.json")):
        try:
            d = json.loads(f.read_text())
            d["_file"] = str(f)
            records.append(d)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f, exc)
    logger.info("Loaded %d full_eval.json files from %s", len(records), input_dir)
    return records


def _ms(vals: list[float]) -> str:
    if not vals:
        return "—"
    return f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}"


def build_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for _key, src, eps_or_key, label in METHODS:
        l1s, rhos, errs = [], [], []
        for r in records:
            v = r.get(src, {}).get(eps_or_key, {})
            if not v or "error" in v:
                continue
            l1s.append(v.get("l1_to_global_mean", float("nan")))
            rhos.append(v.get("spearman_rho_mean", float("nan")))
            errs.append(v.get("adi_error", float("nan")))

        l1s  = [x for x in l1s  if not np.isnan(x)]
        rhos = [x for x in rhos if not np.isnan(x)]
        errs = [x for x in errs if not np.isnan(x)]

        rows.append({
            "Method":           label,
            "L1 mean ↓":        _ms(l1s),
            "Spearman ρ ↑":     _ms(rhos),
            "ADI error mean ↓": _ms(errs),
            "n_cells":          len(l1s),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Directory containing seed_*/alpha_*/full_eval.json files "
                             "(e.g. results/raw/compas)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g. results/tables/table1_compas_aggregate.csv)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    records = load_cells(input_dir)
    if not records:
        logger.error("No full_eval.json files found in %s. "
                     "Extract compas_raw_results.zip there first.", input_dir)
        sys.exit(1)

    df = build_table(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Table 1 written to %s", output_path)

    print("\n=== SoftwareX Table 1 — COMPAS aggregate (all 15 cells) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
