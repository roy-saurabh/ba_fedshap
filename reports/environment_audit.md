# Environment Audit Report
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22
**Platform:** macOS 26.5 (arm64, Apple Silicon)

---

## 1. Required vs Available Versions

| Package | Manuscript-Pinned | Available (system) | Status |
|---------|-------------------|--------------------|--------|
| Python  | 3.10.12           | 3.13.5 (system); 3.12.8 (conda base) | MISMATCH — pinned env required |
| torch   | 2.1.0             | not installed       | MISSING |
| shap    | 0.44.0            | not installed       | MISSING |
| flwr    | 1.5.0             | not installed       | MISSING |
| numpy   | >=1.24            | 2.3.1               | OK |
| pandas  | >=2.0             | 2.3.0               | OK |
| scikit-learn | >=1.3        | 1.7.0               | OK |
| scipy   | >=1.10            | 1.16.0              | OK |
| matplotlib | >=3.7          | 3.10.9              | OK |
| folktables | >=0.0.12       | not installed       | MISSING (ACS datasets) |
| fairlearn  | >=0.10         | not installed       | MISSING (optional) |
| aif360     | >=0.5          | not installed       | MISSING (optional) |

**Hardware (this machine, NOT the paper hardware):**
- CPU: Apple M-series arm64, 14 physical / 14 logical cores
- RAM: 24.0 GB
- GPU: none (Apple Silicon integrated; no CUDA)
- OS: macOS 26.5 arm64

**Paper hardware (as stated in manuscript, Table III):**
- 1× NVIDIA A100 80 GB GPU
- 64-core CPU server

---

## 2. Environment Files Produced

| File | Status | Notes |
|------|--------|-------|
| `environment.yml` | Created | Pins python=3.10.12, torch==2.1.0, shap==0.44.0, flwr==1.5.0 |
| `requirements.txt` | Created | pip-equivalent pins |
| `pyproject.toml`  | Created | PEP 517 package metadata |

---

## 3. Changes Made During Audit

### 3.1 Sigma value correction (CRITICAL — configs updated)

The initial `configs/*.yaml` files contained incorrect `dp_sigmas` values that did not
reproduce the manuscript epsilon targets. Values were recomputed using Theorem 2 of the
manuscript via `src/privacy/rdp_accountant.compute_sigma_for_epsilon`:

Assumptions: T_eff = 40 rounds per client (200 FL rounds × 20% participation per round),
sensitivity Δ₂ = 2C/m_{k,a} = 2×1.0/100 = 0.02, δ = 1e-5.

| ε target | Original σ (wrong) | Corrected σ | Verify: compute_epsilon(T=40, σ, Δ₂=0.02, δ=1e-5) |
|----------|--------------------|-------------|-----------------------------------------------------|
| ∞        | 0.0                | 0.0         | ∞ |
| 8        | 0.018              | 0.0848      | ≈ 8.0 |
| 4        | 0.035              | 0.1583      | ≈ 4.0 |
| 2        | 0.071              | 0.3038      | ≈ 2.0 |
| 1        | 0.141              | 0.5942      | ≈ 1.0 |

All six config files updated: adult.yaml, compas.yaml, german.yaml, bank.yaml,
acs_income.yaml, acs_pubcov.yaml.

### 3.2 Adult feature dimension correction (CRITICAL — src/data/loaders.py updated)

Initial loader used `pd.get_dummies` on all categorical columns, producing d=104.
The manuscript reports d=14 (Table I). The loader was updated to use OrdinalEncoder
for 6 categorical columns (workclass, marital-status, occupation, relationship, race,
sex), 6 continuous features (age, education-num, capital-gain, capital-loss,
hours-per-week, fnlwgt), 1 education ordinal, and 1 binary native-country indicator,
yielding exactly d=14.

Post-fix verification: load_adult() returns X shape (45222, 14), dtype float32,
min=0.0, max=1.0 — matches manuscript Table I.

---

## 4. Import and Smoke-Test Results

```
Dry smoke test (scripts/run_dry_smoke_test.py):
  optional_imports:       PASS  (missing torch/shap/flwr noted but not a failure)
  core_deps:              PASS  (numpy, pandas, scipy, sklearn, matplotlib)
  adult_data_loading:     PASS  (N=45222, d=14 after correction)
  dirichlet_partition:    PASS
  config_loading:         PASS  (all 6 configs, all required fields present)
  adi_norm_metrics:       PASS  (ADI_norm, bootstrap CI, permutation p-value)
  rdp_accountant:         PASS  (ε(σ=0.5942, T=40) ≈ 1.0)
  dp_noise:               PASS  (clip_l2, noisy_mean)
  faithfulness_metrics:   PASS  (deletion_auc, insertion_auc)
  lipschitz_estimator:    SKIP  (requires shap)
  fedavg_training:        SKIP  (requires torch)
  kernelshap_computation: SKIP  (requires torch + shap)
  ba_fedshap_protocol:    SKIP  (requires torch + shap)
  flwr_import:            SKIP  (requires flwr)

Result: PASS (9/9 non-skipped checks), 5 skipped
```

---

## 5. How to Create the Pinned Environment

```bash
# Step 1: Create conda environment
conda env create -f environment.yml
conda activate ba-fedshap

# Step 2: Verify versions
python --version
# Expected: Python 3.10.12
python -c "import torch; print(torch.__version__)"
# Expected: 2.1.0
python -c "import shap; print(shap.__version__)"
# Expected: 0.44.0
python -c "import flwr; print(flwr.__version__)"
# Expected: 1.5.0

# Step 3: Run import checks for all required packages
python -c "import torch, shap, flwr, numpy, pandas, sklearn, scipy, matplotlib, folktables, fairlearn"

# Step 4: Run full smoke test (requires pinned env)
python scripts/run_smoke_test.py
```

---

## 6. Verification Checklist

- [x] `environment.yml` exists and pins all required packages
- [x] `requirements.txt` exists with pip-equivalent pins
- [x] `pyproject.toml` exists with package metadata
- [ ] `conda env create -f environment.yml` completes without errors (requires Python 3.10.12 conda channel)
- [ ] `python -c "import torch; print(torch.__version__)"` prints `2.1.0`
- [ ] `python -c "import shap; print(shap.__version__)"` prints `0.44.0`
- [ ] `python -c "import flwr; print(flwr.__version__)"` prints `1.5.0`
- [x] Dry smoke test passes: `python scripts/run_dry_smoke_test.py`
- [ ] Full smoke test passes: `python scripts/run_smoke_test.py` (requires pinned env)

---

## 7. Verdict

**ENVIRONMENT: PARTIALLY VERIFIED**

Pure-Python components (data loading with corrected d=14, partitioning, ADI metrics,
RDP accounting, DP noise, faithfulness metrics) pass all checks on the current system.
The full pipeline (FedAvg training, KernelSHAP, BA-FedSHAP protocol) requires the
pinned conda environment with Python 3.10.12, torch 2.1.0, shap 0.44.0, flwr 1.5.0,
which is not available on this machine (macOS/arm64).

Two critical corrections were made during this audit: sigma values in all config files
and the Adult dataset feature dimension. These corrections must be reflected in the
final manuscript if the paper is rerun on the pinned hardware.
