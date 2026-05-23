# BA-FedSHAP: Reproducibility Package
**BA-FedSHAP: Background-Anchored Federated Shapley Attributions for Auditable AI**
IEEE Access Submission — Reproducibility Package

---

## Repository Layout

```
BA_FedSHAP-master/
├── paper/
│   └── BA_FedSHAP_v2_IEEEAccess.md      # Manuscript source
├── configs/
│   ├── adult.yaml                         # Adult dataset experiment config
│   ├── compas.yaml                        # COMPAS dataset config
│   ├── german.yaml                        # German Credit config
│   ├── bank.yaml                          # Bank Marketing config
│   ├── acs_income.yaml                    # ACSIncome (Folktables) config
│   └── acs_pubcov.yaml                    # ACSPublicCoverage config
├── src/
│   ├── data/
│   │   ├── loaders.py                     # Dataset loading + preprocessing (d-exact)
│   │   └── partition.py                   # Dirichlet + geography partitioning
│   ├── federated/
│   │   ├── ba_fedshap.py                  # Main protocol: Alg 1, 2, 3
│   │   └── training.py                    # FedAvg trainer, local_train, models
│   ├── metrics/
│   │   ├── adi.py                         # ADI_norm, bootstrap CI, permutation p-value
│   │   ├── faithfulness.py                # Deletion/insertion AUC
│   │   ├── sanity.py                      # Parameter/label randomization checks
│   │   └── lipschitz.py                   # Local Lipschitz estimation
│   ├── explainers/
│   │   ├── kernelshap_wrapper.py          # KernelSHAP with M=2048 coalitions
│   │   └── baselines.py                   # 5 comparison baselines
│   └── privacy/
│       ├── rdp_accountant.py              # RDP accountant (Theorem 2)
│       └── dp_noise.py                    # L2 clipping + Gaussian noise
├── scripts/
│   ├── run_all_main.sh                    # Full pipeline (Tables IV–X)
│   ├── run_experiment.py                  # Single-dataset experiment driver
│   ├── run_ablation.py                    # Table VI ablation sweep
│   ├── run_dp_sweep.py                    # Table VII DP noise sensitivity
│   ├── run_faithfulness.py                # Table V faithfulness/sanity
│   ├── run_stability.py                   # Lipschitz stability (Table IX)
│   ├── make_tables.py                     # Aggregates results → LaTeX/CSV tables
│   ├── make_figures.py                    # Generates paper figures
│   ├── run_smoke_test.py                  # Full smoke test (requires torch/shap/flwr)
│   └── run_dry_smoke_test.py              # Pure-Python checks (no torch required)
├── results/
│   ├── manifest_sha256.json               # SHA256 manifest of result files
│   ├── manifest_sha256.txt                # Human-readable manifest
│   ├── smoke_test/                        # Dry smoke test outputs
│   ├── processed/                         # Per-seed experiment outputs (after full run)
│   └── tables/                            # Final CSV tables (after full run)
├── reports/
│   ├── environment_audit.md               # Environment + sigma correction audit
│   ├── dataset_preprocessing_audit.md     # Per-dataset preprocessing chain
│   ├── acs_audit_split_verification.md    # Folktables audit split verification
│   ├── baseline_audit.md                  # All 10 baseline configurations
│   ├── statistical_validation.md          # Statistical method verification
│   ├── manuscript_table_diff.md           # Table-by-table comparison report
│   ├── manuscript_table_diff.json         # Machine-readable diff (68 entries)
│   ├── reference_audit.md                 # Bibliography validation
│   ├── runtime_hardware_report.md         # Hardware + timing documentation
│   └── hash_manifest_report.md            # Source file SHA256 hashes
├── environment.yml                        # Conda environment (pinned)
├── requirements.txt                       # pip requirements (pinned)
└── pyproject.toml                         # PEP 517 package metadata
```

---

## Hardware Used for Paper Results

| Component | Specification |
|-----------|---------------|
| GPU | 1× NVIDIA A100 80 GB |
| CPU | 64-core server |
| Python | 3.10.12 |
| PyTorch | 2.1.0+cu118 |
| CUDA | ≥ 11.8 |
| shap | 0.44.0 |
| flwr (Flower) | 1.5.0 |

All Table IV–X values in the manuscript were produced on this hardware.
Re-runs on different hardware will produce identical point estimates (fixed seeds)
but runtime values (Table X) will differ.

---

## Python Environment Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ba-fedshap
pip install -e .
```

### Option B: pip

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Required Python version:** 3.10.12 (other versions not tested).

### Verify installation

```bash
python -c "import torch, shap, flwr; print(torch.__version__, shap.__version__, flwr.__version__)"
# Expected: 2.1.0  0.44.0  1.5.0
```

---

## Dataset Download Instructions

### Adult (UCI)

```bash
mkdir -p data/raw/adult
wget -P data/raw/adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -P data/raw/adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

After standard missing-value removal (`?`), N=45,222, d=14.

### COMPAS (ProPublica)

```bash
mkdir -p data/raw/compas
wget -O data/raw/compas/compas-scores-two-years.csv \
  https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

After preprocessing, N=6,172, d=13.

### German Credit (UCI)

```bash
mkdir -p data/raw/german
wget -P data/raw/german https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
```

After preprocessing, N=1,000, d=20.

### Bank Marketing (UCI)

```bash
mkdir -p data/raw/bank
wget -P data/raw/bank https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
unzip data/raw/bank/bank-additional.zip -d data/raw/bank/
```

File used: `bank-additional/bank-additional-full.csv`. After preprocessing, N=41,188, d=17.

### ACSIncome and ACSPublicCoverage (Folktables)

```bash
pip install folktables
# Data is downloaded automatically by loaders.py on first call
# Cache location: ~/.folktables/ (configurable via FOLKTABLES_CACHE env var)
```

Surveys: ACS 2018 1-year, all 50 US states + DC.

---

## Reproduction Commands by Table

All commands assume the conda environment is active and the working directory is the repository root.

### Step 0: Run smoke test (verify installation, ~5 min)

```bash
python scripts/run_smoke_test.py --dataset adult --n_clients 4 --n_rounds 2 --m_coalitions 64
```

Expected output: all checks PASS.

### Step 0b: Dry smoke test (no torch required, ~2 sec)

```bash
python scripts/run_dry_smoke_test.py
```

Expected output: 9 checks PASS, 5 SKIP (torch-dependent).

### Table IV — Primary Reliability (main result)

```bash
# Single dataset, all seeds:
python scripts/run_experiment.py --config configs/adult.yaml --seeds 42 123 456 789 1024
python scripts/run_experiment.py --config configs/compas.yaml --seeds 42 123 456 789 1024
python scripts/run_experiment.py --config configs/german.yaml --seeds 42 123 456 789 1024

# Aggregate into Table IV:
python scripts/make_tables.py --table iv
```

Output: `results/tables/table_iv_primary_reliability.csv`

### Table V — Faithfulness and Sanity

```bash
python scripts/run_faithfulness.py --config configs/adult.yaml --seeds 42 123 456 789 1024
python scripts/make_tables.py --table v
```

Output: `results/tables/table_v_faithfulness_sanity.csv`

### Table VI — Ablation Study

```bash
python scripts/run_ablation.py --config configs/adult.yaml --seeds 42 123 456 789 1024
python scripts/make_tables.py --table vi
```

Output: `results/tables/table_vi_ablation.csv`

### Table VII — DP Noise Sensitivity

```bash
python scripts/run_dp_sweep.py --config configs/adult.yaml \
    --epsilons inf 8 4 2 1 --seeds 42 123 456 789 1024
python scripts/make_tables.py --table vii
```

Output: `results/tables/table_vii_dp_noise.csv`

Note: Corrected sigma values are in `configs/adult.yaml` (`dp_sigmas` field).
These differ from the original submission (see `reports/environment_audit.md` §3.1).

### Table VIII — ADI-Outcome Correlation

```bash
# Requires Tables IV results to be complete first.
python scripts/make_tables.py --table viii
```

Output: `results/tables/table_viii_adi_outcome_correlation.csv`

### Table IX — Lipschitz Stability Estimates

```bash
python scripts/run_stability.py --config configs/adult.yaml --seeds 42 123 456 789 1024
python scripts/make_tables.py --table ix
```

Output: `results/tables/table_ix_lipschitz.csv`

### Table X — Runtime

```bash
# Timings are logged automatically during run_experiment.py.
# After all experiments complete:
python scripts/make_tables.py --table x
```

Output: `results/tables/table_x_runtime.csv`

### Full Pipeline (all tables, all datasets, all seeds)

```bash
bash scripts/run_all_main.sh
```

Estimated wall-clock time on A100 hardware: ~104 hours (all 6 datasets × 5 seeds).
See `reports/runtime_hardware_report.md` §4 for per-dataset estimates.

---

## Expected Runtimes (Paper Hardware: A100 80 GB)

| Dataset | N | d | Estimated wall-clock |
|---------|---|---|----------------------|
| Adult | 45,222 | 14 | 252 min (measured) |
| COMPAS | 6,172 | 13 | ~35 min |
| German Credit | 1,000 | 20 | ~6 min |
| Bank Marketing | 41,188 | 17 | ~230 min |
| ACSIncome | ~195,000 | 10 | ~270 min |
| ACSPublicCoverage | ~109,000 | 19 | ~450 min |
| **Full run (all × 5 seeds)** | — | — | **~104 hours** |

CPU-only re-runs will be 10–50× slower due to KernelSHAP (M=2048 coalitions per client).
Reduce `m_coalitions` to 64 for faster smoke testing (results will differ from manuscript).

---

## Pre-registered Seeds

Seeds used for all experiments: **42, 123, 456, 789, 1024**

These were fixed before any experiments were run. Do not change them for reproduction.

---

## SHA256 Manifest

Source file hashes are documented in `reports/hash_manifest_report.md`.

`results/manifest_sha256.json` covers the files currently present in the repo:
`results/processed/manuscript_values.json` and the seven table CSVs.
Raw per-cell `full_eval.json` files for the 13 executed cells are archived on
Zenodo at https://doi.org/10.5281/zenodo.20356218.

**Note on compas_raw_results.zip:** The GitHub release asset
(`v1.0.0-softx/compas_raw_results.zip`) has been removed. It contained only
two schema-incompatible partial result files for already-executed cells
(seed=789 and seed=1024 at α=0.50) and did not represent the 13-cell corpus.
The authoritative raw-results archive is the Zenodo deposit above.

To regenerate the result manifest after a full run:

```bash
python3 - <<'PYEOF'
import hashlib, json
from pathlib import Path

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

manifest = {}
for d in ["results/processed", "results/tables"]:
    for f in sorted(Path(d).rglob("*.csv")):
        manifest[str(f)] = {"sha256": sha256(f), "size_bytes": f.stat().st_size}

with open("results/manifest_sha256.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest written: {len(manifest)} files")
PYEOF
```

---

## Known Limitations and Discrepancies

### L1: Full pipeline not executed on audit machine

The audit machine (Apple M-series, no CUDA, Python 3.13.5) cannot run the pinned
environment. All 68 table values in `reports/manuscript_table_diff.md` are
GENERATED_VALUE_MISSING. Verification requires the pinned conda environment on
CUDA hardware.

### L2: DP sigma values corrected

The original config files used sigma values (0.018, 0.035, 0.071, 0.141) that do not
correspond to the claimed epsilon targets (8, 4, 2, 1) under Theorem 2's RDP accountant
with T=40 rounds, Δ₂=0.02, δ=1e-5.

Corrected values (all configs updated):

| ε | σ (original) | σ (corrected) |
|---|-------------|---------------|
| 8 | 0.018 | 0.0848 |
| 4 | 0.035 | 0.1583 |
| 2 | 0.071 | 0.3038 |
| 1 | 0.141 | 0.5942 |

If the manuscript was run with original sigma values, Tables IV and VII will differ
from a re-run with corrected values. See `reports/environment_audit.md` §3.1.

### L3: Adult sample count

Manuscript Table I reports N=48,842 for Adult. After standard missing-value removal,
the actual sample count is N=45,222. The manuscript should be updated.

### L4: Fisher z CI discrepancy (Table VIII)

Manual calculation for Adult r=0.71, n=25 gives CI=[0.44, 0.86] using the standard
Fisher z-transform. The manuscript reports [0.52, 0.83]. This discrepancy requires
investigation; see `reports/statistical_validation.md` §3.

### L5: Missing Table II

The manuscript references "Costs are summarized in Table II" but no numbered Table II
appears in the manuscript. This must be added or the reference removed before submission.

### L6: ACS datasets require folktables

ACSIncome and ACSPublicCoverage are downloaded automatically by `loaders.py` via the
`folktables` package. An internet connection is required on first run.

---

## How to Verify Manuscript Tables Match Generated Tables

After the full pipeline runs:

```bash
# Generate tables from results:
python scripts/make_tables.py

# Compare generated tables against manuscript values:
python scripts/compare_manuscript.py

# Outputs:
#   reports/manuscript_table_diff.md  (human-readable)
#   reports/manuscript_table_diff.json (machine-readable, 68 entries)
```

The comparison script checks each cell against the manuscript value with a 1% relative
tolerance for floating-point rounding. Any MISMATCH is reported with the manuscript
value, the generated value, and the relative error.

---

## Smoke Test Procedure

**Quick (no torch required):**
```bash
python scripts/run_dry_smoke_test.py
# Expected: PASS 9/9 non-torch checks
```

**Full (requires pinned environment):**
```bash
python scripts/run_smoke_test.py --dataset adult --n_clients 4 --n_rounds 2 --m_coalitions 64
# Expected: all checks PASS, runtime ~5 min on GPU
```

Both produce output to `results/smoke_test/`.

---

## Contact and Citation

For reproducibility questions related to this submission, see the audit reports in `reports/`.

Pre-registered seeds, sigma correction rationale, and all audit findings are documented
in the reports directory and must be disclosed in the manuscript's reproducibility statement.

**Zenodo archive:** https://doi.org/10.5281/zenodo.20356218

To cite this software:

```
Roy Saurabh. BA-FedSHAP: Background-Anchored Federated Shapley Attributions for Auditable AI.
Zenodo, 2025. https://doi.org/10.5281/zenodo.20356218
```
