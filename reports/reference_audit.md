# Reference Audit Report
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22

---

## 1. Reference Numbering Gaps

IEEE Access requires sequential reference numbering. The manuscript has the following gaps:

Missing numbers: [31], [32], [38], [40], [41], [42], [45], [46], [49]

Present numbers: [1]-[30], [33]-[37], [39], [43]-[44], [47]-[48], [50]-[59]

**Required action:** Renumber all references sequentially. This is a formatting
requirement for IEEE Access submission.

---

## 2. Problematic References (Implementation Descriptions in Bibliography)

The following three reference entries are internal implementation descriptions, not
published works. They MUST be moved out of the bibliography before submission.

### Reference [10]

> [10] M. Wang and W. Deng, "Federated SHAP via background synthesis (re-implementation),"
> implementation variant used as a baseline in this paper; configuration in
> configs/baselines/kmeans_background.yaml in the reproducibility package.

**Issue:** This is not a published paper. It describes an internal implementation.
The phrase "re-implementation" and "implementation variant used as a baseline in this
paper" explicitly marks it as in-house code, not a citable work.

**Required action:** Remove [10] from the bibliography. Move the implementation
description to the README or supplementary methods section. In the manuscript body,
replace the citation with a footnote or inline parenthetical describing the baseline
construction.

### Reference [15]

> [15] Y. Wang, J. Liang, and Q. Yang, "Distributed and collaborative SHAP
> (DC-SHAP, re-implementation)," implementation variant used as a baseline;
> configuration in configs/baselines/shared_background.yaml.

**Issue:** Same as [10]. Not a published work. Contains "re-implementation" and
"implementation variant used as a baseline."

**Required action:** Remove [15] from bibliography. Replace with inline description
or footnote. If "DC-SHAP" has a real published source, cite that instead.

### Reference [20]

> [20] L. Zhang, B. Cui, and Z. Jia, "FedXAI: Communication-efficient federated
> explainability via gradient-based attribution (re-implementation)," implementation
> variant used as a baseline; configuration in configs/baselines/gradient_federated.yaml.

**Issue:** Same as [10] and [15]. Not a published work.

**Required action:** Remove [20] from bibliography. The gradient-based attribution
baseline is based on Sundararajan et al. [13] (Integrated Gradients), which is
properly cited. Replace [20] with a reference to [13] or the actual published FedXAI
paper if it exists as a conference/journal paper.

---

## 3. Reference Validity Audit

All other references are real published works, preprints, software packages, or datasets.

| Ref | Title/Source | Type | Status |
|-----|-------------|------|--------|
| [1] | Kairouz et al., Advances and open problems in FL, FnTML 2021 | Journal | VALID |
| [2] | Lundberg & Lee, SHAP, NeurIPS 2017 | Conference | VALID |
| [3] | McMahan et al., FedAvg, AISTATS 2017 | Conference | VALID |
| [4] | Bonawitz et al., Secure aggregation, CCS 2017 | Conference | VALID |
| [5] | Abadi et al., DP-SGD, CCS 2016 | Conference | VALID |
| [6] | Lundberg et al., TreeSHAP, Nature MI 2020 | Journal | VALID |
| [7] | Jethani et al., FastSHAP, ICLR 2022 | Conference | VALID |
| [8] | Frye, Feige & Rowat, Asymmetric Shapley, NeurIPS 2020 | Conference | VALID |
| [9] | Li et al., FL challenges, IEEE SPM 2020 | Journal | VALID |
| [10] | "re-implementation" | INTERNAL — REMOVE | INVALID |
| [11] | Li et al., FedProx, MLSys 2020 | Conference | VALID |
| [12] | Karimireddy et al., SCAFFOLD, ICML 2020 | Conference | VALID |
| [13] | Sundararajan et al., Integrated Gradients, ICML 2017 | Conference | VALID |
| [14] | Ribeiro et al., LIME, KDD 2016 | Conference | VALID |
| [15] | "DC-SHAP, re-implementation" | INTERNAL — REMOVE | INVALID |
| [16] | Shokri et al., Privacy risks of explanations, AIES 2021 | Conference | VALID |
| [17] | Patel et al., DP explanations, SPW 2022 | Workshop | VALID |
| [18] | Beutel et al., Flower, arXiv 2020 | Preprint/Software | VALID |
| [19] | Lundberg, SHAP Python Package v0.44.0 | Software | VALID |
| [20] | "FedXAI, re-implementation" | INTERNAL — REMOVE | INVALID |
| [21] | Lloyd, k-means, IEEE TIT 1982 | Journal | VALID |
| [22] | El Mhamdi et al., Byzantine in Byzantium, ICML 2018 | Conference | VALID |
| [23] | Cao et al., FLTrust, NDSS 2021 | Conference | VALID |
| [24] | Covert et al., Explaining by removing, JMLR 2021 | Journal | VALID |
| [25] | Dwork & Roth, Algorithmic foundations of DP, FnTCS 2014 | Monograph | VALID |
| [26] | Villani, Optimal Transport, Springer 2008 | Book | VALID |
| [27] | Fournier & Guillin, Wasserstein rates, PTRF 2015 | Journal | VALID |
| [28] | Iglewicz & Hoaglin, Detect and Handle Outliers, ASQ 1993 | Book | VALID |
| [29] | Ghorbani & Zou, Data Shapley, ICML 2019 | Conference | VALID |
| [30] | Bellamy et al., AI Fairness 360, IBM JRD 2019 | Journal | VALID |
| [33] | Cohen, Statistical Power Analysis, LEA 1988 | Book | VALID |
| [34] | Cliff, Dominance statistics, Psych Bull 1993 | Journal | VALID |
| [35] | Kohavi, UCI Adult dataset, KDD 1996 | Conference+Dataset | VALID |
| [36] | Angwin et al., Machine Bias, ProPublica 2016 | Journalism+Dataset | VALID |
| [37] | Dua & Graff, UCI ML Repository, UCI 2017 | Repository | VALID |
| [39] | Hardt et al., Equal opportunity, NeurIPS 2016 | Conference | VALID |
| [43] | Weerts et al., Fairlearn, JMLR 2023 | Journal | VALID |
| [44] | Mironov, RDP, CSF 2017 | Conference | VALID |
| [47] | Mitchell et al., Model cards, FAccT 2019 | Conference | VALID |
| [48] | Doshi-Velez & Kim, Rigorous IML, arXiv 2017 | Preprint | VALID |
| [50] | Benjamini & Yekutieli, FDR, Ann Stat 2001 | Journal | VALID |
| [51] | Agarwal et al., Reductions for fair classification, ICML 2018 | Conference | VALID |
| [52] | Covert et al., Explaining by removing, JMLR 2021 (duplicate of [24]) | Journal | NOTE: duplicate |
| [53] | Ding et al., Retiring Adult, NeurIPS 2021 (folktables) | Conference+Dataset | VALID |
| [54] | Adebayo et al., Sanity checks, NeurIPS 2018 | Conference | VALID |
| [55] | Hooker et al., ROAR, NeurIPS 2019 | Conference | VALID |
| [56] | Rong et al., ROAD, ICML 2022 | Conference | VALID |
| [57] | Slack et al., Fooling LIME and SHAP, AIES 2020 | Conference | VALID |
| [58] | Heskes et al., Causal Shapley, NeurIPS 2020 | Conference | VALID |
| [59] | Owen & Prieur, Shapley for dependent inputs, SIAM 2017 | Journal | VALID |

---

## 4. Duplicate References

**[24] and [52] are identical:**
Both cite Covert, Lundberg & Lee "Explaining by removing" JMLR 2021.
One of these must be removed and all in-text citations updated to point to the
remaining entry.

---

## 5. Missing References

The manuscript text references the following concepts without a specific citation:
- "Rényi composition" (cites [44] Mironov — ADEQUATE)
- "post-processing closure" [25] — ADEQUATE
- "Iglewicz and Hoaglin" [28] — ADEQUATE

No missing citations identified beyond the internal references flagged above.

---

## 6. "This submission" Language

The following occurrences of "this submission" should be changed to "this paper"
or "this work" for the final IEEE Access version:

- Section VI.A line: "FEMNIST and CelebA are excluded from this submission"
- Section IV comments: "In this submission, backgrounds are constructed from..."
- Section VIII.A: "The privacy guarantee in Theorem 2 is the only privacy guarantee
  claimed in this paper" (already uses "paper" — OK)

---

## 7. Summary

| Category | Count | Status |
|----------|-------|--------|
| Valid references | 52 | OK |
| INVALID (internal implementation descriptions) | 3 | MUST REMOVE: [10], [15], [20] |
| Duplicate references | 1 pair | MUST FIX: [24]/[52] |
| Numbering gaps | 9 | MUST RENUMBER before submission |
| "this submission" language | 2 occurrences | SHOULD FIX |

---

## 8. Required Actions Before Submission

1. Remove references [10], [15], [20] from bibliography
2. Move their content to README.md supplementary methods section
3. Renumber all remaining references sequentially (1 through ~53)
4. Update all in-text citations to reflect new numbering
5. Merge [24] and [52] (same paper cited twice)
6. Replace "this submission" with "this paper" or "this work" throughout
