# Manuscript Table Diff Report
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22

---

## Executive Summary

The full empirical pipeline has NOT been executed on this machine because the pinned
environment (Python 3.10.12, torch==2.1.0, shap==0.44.0, flwr==1.5.0) is unavailable
and the paper hardware (1x NVIDIA A100 80 GB) is not present.

All 68 manuscript table values are therefore classified as GENERATED_VALUE_MISSING.

Once the pipeline is executed on the pinned hardware, this report must be regenerated
by running:

    python scripts/make_tables.py && python scripts/compare_manuscript.py

and the diff.json re-parsed.

| Status | Count |
|--------|-------|
| MATCH_EXACT | 0 |
| MATCH_ROUNDED | 0 |
| MISMATCH | 0 |
| MANUSCRIPT_VALUE_MISSING | 0 |
| GENERATED_VALUE_MISSING | 68 |

---

## Pre-Run Known Discrepancies

These discrepancies are identifiable from code inspection alone, before execution.

### D1: DP sigma values (Tables IV, VII)

The manuscript Table VII reports DP sensitivity sweep at eps in {inf, 8, 4, 2, 1}.
The original config sigma values (0.018, 0.035, 0.071, 0.141) did NOT correspond to
those epsilon values under the Theorem 2 RDP accountant.

Corrected sigma values (environment_audit.md section 3.1):
- eps=8: sigma=0.0848 (was 0.018)
- eps=4: sigma=0.1583 (was 0.035)
- eps=2: sigma=0.3038 (was 0.071)
- eps=1: sigma=0.5942 (was 0.141)

Impact: If the manuscript was run with the original (incorrect) sigma values, the
re-run with corrected sigmas will produce different DP-noise results in Tables IV
and VII. The corrected values represent the true epsilon targets. This is a
MISMATCH that cannot be resolved by adjusting the code — it reflects a correction
to the experiment configuration.

### D2: Adult sample count (Table I, implicitly Tables IV-X)

Manuscript Table I reports N=48,842 for Adult. After standard preprocessing
(dropping rows with missing values "?"), the actual N=45,222. This is consistent
with published SHAP/fairness papers and the UCI documentation which notes ~7% missing
values. The manuscript should be updated to report N=45,222 or clarify "before
missing-value removal."

### D3: Fisher z CI widths (Table VIII)

Manual calculation for Adult r=0.71, n=25 gives CI=[0.44, 0.86] using the standard
Fisher z transform. The manuscript reports [0.52, 0.83]. These do not match.
This discrepancy is classified as MISMATCH once generated values are available.

Possible explanations:
1. The paper uses a larger effective n (pooling more configurations)
2. The paper uses a different CI method (bias-corrected bootstrap)
3. The n=25 interpretation is incorrect

This must be resolved before submission.

---

## Table-by-Table Status

### Table IV: Primary Reliability

All 28 values: GENERATED_VALUE_MISSING (requires full pipeline run).

Expected values from manuscript (5 seeds x 3 tabular datasets, Dir(alpha=0.5)):

| Method | L1 | Spearman rho | |ADI-oracle| | 95% CI covers |
|--------|----|-------------|-------------|-------------|
| Local SHAP | 0.187 +/- .041 | 0.721 +/- .063 | 0.062 [.049,.075] | 0.32 |
| Naive aggregated | 0.153 +/- .038 | 0.769 +/- .057 | 0.049 [.039,.060] | 0.48 |
| Shared-background | 0.097 +/- .026 | 0.852 +/- .043 | 0.024 [.018,.031] | 0.76 |
| k-means background | 0.089 +/- .024 | 0.863 +/- .041 | 0.021 [.016,.028] | 0.80 |
| Gradient-based | 0.104 +/- .027 | 0.843 +/- .045 | 0.027 [.020,.034] | 0.72 |
| BA-FedSHAP (inf) | 0.061 +/- .017 | 0.921 +/- .029 | 0.011 [.007,.016] | 0.92 |
| BA-FedSHAP (eps=4) | 0.074 +/- .021 | 0.904 +/- .034 | 0.014 [.009,.020] | 0.88 |
| BA-FedSHAP (eps=1) | 0.098 +/- .028 | 0.879 +/- .041 | 0.020 [.013,.028] | 0.80 |

### Table V: Faithfulness and Sanity

All 16 values: GENERATED_VALUE_MISSING.

### Table VI: Ablation

All 10 values: GENERATED_VALUE_MISSING.

### Table VII: DP Noise Sensitivity

All 10 values: GENERATED_VALUE_MISSING.
Note: corrected sigma values will change results vs. manuscript (see D1 above).

### Table VIII: ADI-Outcome Correlation

All 12 values: GENERATED_VALUE_MISSING.
Note: Fisher z CI discrepancy (see D3 above).

### Table IX: Lipschitz Estimates

All 7 values: GENERATED_VALUE_MISSING.

### Table X: Runtime

All 5 values: GENERATED_VALUE_MISSING.
Note: Timings were reported for A100 GPU. Re-run on different hardware will differ.

---

## Tables Referenced but Missing from Manuscript

### Table II: Complexity Summary

The manuscript text at Section IV ("Costs are summarized in Table II") references
Table II, but Table II does not appear in the manuscript as a numbered table. The
text describes per-operation costs inline. This creates a forward-reference to a
non-existent numbered table.

**Required action:** Either add Table II with the complexity summary, or change the
reference to "described below" / remove the Table II reference.

---

## Action Items for Full Run

1. Set up pinned conda environment (Python 3.10.12, torch 2.1.0, shap 0.44.0, flwr 1.5.0)
2. Run full pipeline: bash scripts/run_all_main.sh
3. Generate tables: python scripts/make_tables.py
4. Re-run this comparison: python scripts/compare_manuscript.py (to be created)
5. Update manuscript with corrected N=45,222 for Adult
6. Investigate and resolve Fisher z CI discrepancy for Table VIII
7. Update manuscript Tables IV/VII if corrected sigma values change results
8. Add missing Table II (complexity summary)
