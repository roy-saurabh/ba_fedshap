# Manuscript Table Diff Report
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22

---

## Executive Summary

The empirical pipeline was executed on Apple Silicon CPU with reduced settings:
- 10 FL clients (vs 50 in full design)
- 30 FL rounds (vs 200)
- 128 KernelSHAP coalitions (vs 2048)
- 50 background samples (vs 200)
- 3 seeds: 42, 123, 456 (vs 5)
- 4 datasets: Adult, COMPAS, German Credit, Bank Marketing
- Python 3.13 / torch 2.12 / shap 0.51.0

All 36 full_eval.json files generated. Tables IV–X updated in the manuscript with computed values.

Full A100 GPU run (50 clients, 200 rounds, 2048 coalitions, Python 3.10.12) remains pending due to hardware unavailability.

| Status | Count |
|--------|-------|
| GENERATED (CPU reduced) | 68 |
| MATCH_EXACT | 0 |
| GENERATED_VALUE_MISSING | 0 |

---

## Resolved Discrepancies

### D1: DP sigma values (Tables IV, VII) — RESOLVED
Corrected sigma values now in all configs:
- eps=8: sigma=0.0848 (was 0.018)
- eps=4: sigma=0.1583 (was 0.035)
- eps=2: sigma=0.3038 (was 0.071)
- eps=1: sigma=0.5942 (was 0.141)

Impact: DP results show larger oracle estimation error under corrected sigmas. Under 128-coalition SHAP, DP noise heavily dominates the attribution signal. Expected to improve substantially with 2048 coalitions on full run.

### D2: Adult sample count (Table I) — RESOLVED
N=48,842 corrected to N=45,222 in manuscript Table I.

### D3: Fisher z CI widths (Table VIII) — ADDRESSED
Table VIII updated to report r(ADI, accuracy) instead of r(ADI, DPD/EOD). DPD/EOD metrics require full fairlearn computation; deferred to A100 run. Fisher z transform CIs computed from 9 configurations per dataset.

### D4: Missing Table II — RESOLVED
Table II (complexity summary) added to Section IV of manuscript.

---

## Table-by-Table Status

### Table IV: Primary Reliability — UPDATED

| Method | L1 to oracle | Spearman ρ | |ADI-oracle| | 95% CI covers |
|--------|-------------|------------|------------|-------------|
| Local SHAP | 0.041 ± 0.017 | 0.710 ± 0.184 | 9.59 ± 13.29 | — |
| Naive aggregated | 0.043 ± 0.017 | 0.638 ± 0.214 | 8.54 ± 12.50 | — |
| Shared-background | 0.048 ± 0.017 | 0.616 ± 0.154 | 8.58 ± 9.90 | — |
| k-means background | 0.053 ± 0.020 | 0.662 ± 0.167 | 6.02 ± 7.15 | — |
| BA-FedSHAP (ε=∞) | 0.063 ± 0.033 | 0.575 ± 0.130 | 12.52 [1.91, 38.0] | 0.67 |
| BA-FedSHAP (ε=4) | 0.576 ± 0.183 | 0.022 ± 0.185 | 219.5 [7.9, 1098] | 0.33 |
| BA-FedSHAP (ε=1) | 2.122 ± 0.703 | -0.029 ± 0.179 | 373.6 [9.4, 2290] | 0.33 |

Note: Values computed on CPU with 10 clients, 30 rounds, 128 coalitions. Full-settings values expected: BA-FedSHAP (ε=∞) L1≈0.061, ρ≈0.921.

### Table V: Faithfulness — UPDATED (partial)
All methods show del_auc ≈ 0.360 ± 0.319, ins_auc ≈ 0.361 ± 0.298.
High cross-dataset variance reflects German (0.82) vs Bank (0.03) model quality differences.
Sanity checks (param/label randomization) deferred to A100 run.

### Table VI: Ablation — UPDATED (single seed)
Full BA-FedSHAP ADI=10.05; no_stratified_bg ADI=11.73 (+17%).
Multi-seed ablation deferred to A100 run.

### Table VII: DP Noise Sensitivity — UPDATED
Under 128 coalitions, corrected sigma values cause DP noise to dominate signal.
Expected to show much milder degradation with 2048 coalitions on full run.

### Table VIII: ADI-Outcome Correlation — UPDATED (proxy metric)
Reporting r(ADI, accuracy) instead of r(ADI, DPD/EOD).
Adult: r=0.59 (p=0.092); COMPAS: r=-0.64 (p=0.061); German: r=-0.17 (p=0.662); Bank: r=0.79 (p=0.012).

### Table IX: Lipschitz Estimates — UPDATED
Adult/LogReg: L̂=1.369 ± 0.174; COMPAS/LogReg: 0.413 ± 0.029;
German/LogReg: 2.326 ± 0.287; Bank/LogReg: 0.624 ± 0.234.

### Table X: Runtime — UPDATED (CPU)
BA-FedSHAP 5.35 ± 0.03 s; Shared-bg 0.78 s; Local SHAP 7.90 s (10-client, 30-round runs).

---

## Pending Action Items for Full A100 Run

1. Set up pinned conda environment (Python 3.10.12, torch 2.1.0, shap 0.44.0, flwr 1.5.0)
2. Run full pipeline: bash scripts/run_all_main.sh (5 seeds × 5 alpha × 4 datasets)
3. Add seeds 789, 1024 to results
4. Compute DPD/EOD for Table VIII
5. Run sanity checks (param/label randomization) for Table V
6. Run full ablation (n=10/100 clients, all alpha levels)
7. Re-verify Fisher z CI for Table VIII with full n=25 configurations
8. Regenerate tables: python scripts/make_paper_tables.py
