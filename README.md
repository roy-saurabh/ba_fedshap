# BA-FedSHAP

**A reproducible toolkit for auditing background-induced attribution drift in federated SHAP explanations**

*SoftwareX submission — v1.0.1-softx*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20356218.svg)](https://doi.org/10.5281/zenodo.20356218)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10/3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)

---

## Overview

BA-FedSHAP is a diagnostic toolkit that measures how the choice of SHAP background dataset affects attribution quality in federated learning settings, under varying degrees of differential privacy (DP) noise.

The toolkit does **not** claim that any federated SHAP variant is superior. It surfaces a reproducible, quantified account of how background anchoring interacts with DP noise, data heterogeneity, and federation scale. The primary metric is Attribution Drift Index (ADI): how much do SHAP attributions shift when the background changes?

Key findings: at moderate-to-strong DP levels (ε ≤ 4), all background strategies produce attributions that diverge substantially from local-data SHAP values. The toolkit makes this degradation measurable and auditable.

---

## Repository layout

```
ba_fedshap/
├── src/
│   ├── data/
│   │   ├── loaders.py              # Dataset loading and preprocessing
│   │   └── partition.py            # Dirichlet + geography partitioning
│   ├── federated/
│   │   ├── ba_fedshap.py           # Main protocol (Algorithms 1–3)
│   │   └── training.py             # FedAvg trainer, local_train, models
│   ├── metrics/
│   │   ├── adi.py                  # ADI_norm, bootstrap CI, permutation p-value
│   │   ├── faithfulness.py         # Deletion/insertion AUC (NumPy-compat)
│   │   ├── sanity.py               # Parameter/label randomisation checks
│   │   └── lipschitz.py            # Local Lipschitz estimation
│   ├── explainers/
│   │   ├── kernelshap_wrapper.py   # KernelSHAP wrapper
│   │   └── baselines.py            # Five comparison baselines
│   └── privacy/
│       ├── rdp_accountant.py       # RDP accountant
│       └── dp_noise.py             # L2 clipping + Gaussian noise
├── configs/
│   ├── compas.yaml                 # Full-scale COMPAS config (5 seeds × 5 α × 5 DP)
│   ├── compas_lowcompute_softx.yaml # SoftwareX low-compute variant (≈2 CPU-hours)
│   ├── adult.yaml
│   ├── bank.yaml
│   ├── german.yaml
│   ├── acs_income.yaml
│   └── acs_pubcov.yaml
├── scripts/
│   ├── run_full_eval.py            # Per-seed, per-alpha experiment driver
│   ├── run_softx_compas_lowcompute.sh  # One-command SoftwareX reproducer
│   ├── run_experiment.py           # Single-dataset driver
│   ├── run_ablation.py             # Ablation sweep
│   ├── run_dp_sweep.py             # DP noise sensitivity sweep
│   ├── run_faithfulness.py         # Faithfulness/sanity metrics
│   ├── run_stability.py            # Lipschitz stability
│   ├── make_tables.py              # Aggregate results → CSV/LaTeX
│   └── make_figures.py             # Generate paper figures
├── results/
│   └── manifest_sha256.json        # SHA-256 manifest (in-repo + Zenodo scope)
├── reports/                        # Audit reports (environment, datasets, statistics)
├── pyproject.toml
├── environment.yml
├── LICENSE
├── LICENSE.txt
└── CITATION.cff
```

---

## Installation

**Python 3.10 or 3.11 required.** Python 3.12 is not supported because the pinned `torch==2.1.0` and `flwr==1.5.0` do not install cleanly on 3.12.

```bash
# Create environment (conda recommended)
conda env create -f environment.yml
conda activate ba_fedshap

# Or pip install
pip install -e ".[dev]"
```

---

## Reproducing the SoftwareX results (COMPAS, low-compute)

The low-compute variant reproduces the manuscript Table 1 on a standard laptop/desktop CPU in approximately 2 hours. It uses:

- `n_clients = 10`, `fl_rounds = 10`
- `kernelshap_coalitions = 64`, `background_size = 200`
- 5 seeds × 3 α levels (0.1, 0.5, 1.0) = **15 cells**
- 5 DP levels: ε ∈ {∞, 8, 4, 2, 1}

**One-command reproducer:**

```bash
bash scripts/run_softx_compas_lowcompute.sh
```

This runs all 15 cells sequentially and writes per-cell `full_eval.json` files to `results/raw/`.

**Or manually, for a single cell:**

```bash
python scripts/run_full_eval.py \
  --config configs/compas_lowcompute_softx.yaml \
  --seed 42 \
  --alpha 0.1 \
  --results-dir results/raw \
  --n-clients 10 \
  --fl-rounds 10 \
  --coalitions 64 \
  --background-size 200 \
  --skip-sanity \
  --skip-lipschitz
```

---

## Pre-computed raw results (Zenodo)

The 15 `full_eval.json` files (5 seeds × 3 α levels) are archived on Zenodo:

**DOI: [10.5281/zenodo.20356218](https://doi.org/10.5281/zenodo.20356218)**

The Zenodo archive contains `compas_raw_results.zip` with all 15 cells, SHA-256:

```
378a6c5b91c2a760aacb6a88b4ebab48b69f3491468375067545f264d88186bb
```

To reproduce Table 1 from the pre-computed raw results without re-running experiments:

```bash
# After downloading and extracting compas_raw_results.zip to results/raw/
python scripts/make_tables.py --results-dir results/raw --output results/tables/
```

---

## Expected results (Table 1)

The manuscript reports the following aggregate means across 15 cells (5 seeds × 3 α):

| Method             | L1 mean | ρ mean | ADI error mean |
|--------------------|--------:|-------:|---------------:|
| Local SHAP         |  0.0217 | 0.7179 |         0.6644 |
| Naive aggregated   |  0.0217 | 0.6956 |         0.6365 |
| Shared background  |  0.0206 | 0.7718 |         0.8829 |
| k-means background |  0.0373 | 0.7993 |         0.7943 |
| BA-FedSHAP ε=∞     |  0.0243 | 0.5029 |        12.5355 |
| BA-FedSHAP ε=8     |  0.3353 | 0.0678 |        92.2070 |
| BA-FedSHAP ε=1     |  2.3597 | 0.0458 |       106.2784 |

These values are verifiable from `compas_aggregate.json` in the Zenodo archive.

---

## Full-scale experiments

The full-scale config (`configs/compas.yaml`) uses `n_clients=50`, `fl_rounds=200`, `kernelshap_coalitions=2048`, and all five α levels. It requires significant compute (GPU recommended). The full raw result corpus is also archived on Zenodo.

---

## Citation

If you use this software, please cite:

```bibtex
@software{saurabh2026bafedshap,
  author  = {Saurabh, Roy},
  title   = {{BA-FedSHAP}: A reproducible toolkit for auditing background-induced
             attribution drift in federated {SHAP} explanations},
  version = {v1.0.1-softx},
  year    = {2026},
  doi     = {10.5281/zenodo.20356218},
  url     = {https://github.com/roy-saurabh/ba_fedshap}
}
```

---

## License

MIT — see [LICENSE](LICENSE) / [LICENSE.txt](LICENSE.txt).
