# Runtime and Hardware Report
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22

---

## 1. Paper Hardware (manuscript Table III)

| Component | Specification |
|-----------|---------------|
| GPU | 1x NVIDIA A100 80 GB |
| CPU | 64-core server |
| Python | 3.10.12 |
| PyTorch | 2.1.0 |
| CUDA | implied >= 11.8 for A100 + torch 2.1.0 |

---

## 2. Audit Machine (this run)

| Component | Specification |
|-----------|---------------|
| CPU | Apple M-series arm64, 14 cores |
| GPU | Apple Silicon integrated (no CUDA) |
| RAM | 24.0 GB |
| OS | macOS 26.5 arm64 |
| Python | 3.13.5 (system); pinned 3.10.12 not available |
| PyTorch | Not installed |
| CUDA | N/A |

The full pipeline was NOT executed on this machine. Timings are from the manuscript.

---

## 3. Manuscript-Reported Runtimes (Table X)

Adult, n=50 clients, m=10 per round, T=200 rounds, single seed, A100 hardware.

| Stage | BA-FedSHAP (eps=inf) | Shared-bg SHAP | Local SHAP |
|-------|---------------------|----------------|------------|
| Background construction (one-time) | 4.1 s | 3.8 s | N/A |
| Client-side SHAP per round (per client) | 21.4 s | 20.9 s | 20.7 s |
| DP clipping + noise (per client) | 0.04 s | N/A | N/A |
| Server-side robust aggregation | 0.31 s | 0.12 s | 0.12 s |
| End-to-end (200 rounds, 50 clients) | 252 min | 240 min | 228 min |

BA-FedSHAP overhead vs shared-background SHAP: ~5% end-to-end.

---

## 4. Estimated Runtimes per Dataset (paper hardware, rough scaling)

| Dataset | N | d | Estimated end-to-end |
|---------|---|---|----------------------|
| Adult | 45,222 | 14 | 252 min (measured) |
| COMPAS | 6,172 | 13 | ~35 min |
| German Credit | 1,000 | 20 | ~6 min |
| Bank Marketing | 41,188 | 17 | ~230 min |
| ACSIncome | ~195,000 | 10 | ~270 min |
| ACSPublicCoverage | ~109,000 | 19 | ~450 min |
| Full run (all datasets x 5 seeds) | — | — | ~104 hours total |

KernelSHAP with M=2048 coalitions dominates all timings.

---

## 5. Dry Smoke Test Timing (this machine)

From results/smoke_test/dry_smoke_test_results.json (pure-Python checks only):

| Check | Time |
|-------|------|
| core_deps | 0.78 s |
| adult_data_loading | 0.20 s (cached) |
| dirichlet_partition | 0.01 s |
| config_loading | 0.02 s |
| adi_norm_metrics | 0.01 s |
| rdp_accountant | 0.00 s |
| dp_noise | 0.00 s |
| faithfulness_metrics | 0.11 s |
| Total (9 checks, torch/shap/flwr skipped) | 1.1 s |

---

## 6. Runtime CSV (placeholder)

results/tables/runtime_by_dataset.csv will be populated from timing logs after the
full run. Its schema:

    dataset, method, stage, wall_clock_s, seed, hardware

---

## 7. Caveats

- Table X timings cannot be independently verified without A100 hardware.
- CPU-only re-runs will be 10-50x slower than A100 for the KernelSHAP step.
- Reducing M from 2048 to 64 gives ~32x speedup for smoke tests.
- All Table X values are GENERATED_VALUE_MISSING in the table diff report.
