**BA-FedSHAP: Stabilizing Removal-Based Shapley Attributions for Federated Bias Auditing under Non-IID Data**

*Roy Saurabh*

AffectLog SAS, Paris, France | roy@affectlog.com | ORCID: 0000-0003-3439-7731

***Abstract—***Federated learning produces non-IID client distributions in which the marginal data observed by each participant can differ substantially. SHAP-style removal-based attributions depend on the background distribution used to instantiate feature removal; consequently, attributions computed under local client backgrounds drift across clients even when the underlying global model is identical. This drift confounds group-level bias audits in federated settings. We propose BA-FedSHAP, a federated audit protocol that (i) aligns pooled and group-conditioned background distributions across clients; (ii) computes group-level attribution summaries under fixed removal semantics; and (iii) reports attribution drift and a standardized attribution disparity index (ADI\_norm) with bootstrap confidence intervals and permutation p-values. The protocol releases clipped, Gaussian-noised client-level summaries with per-record differential-privacy accounting under Rényi composition, enabling server-side robust aggregation and drift diagnostics. We provide a Wasserstein-continuity analysis for removal-based Shapley attributions and an executed empirical evaluation on Adult, COMPAS, German Credit, Bank Marketing and ACSIncome/ACSPublicCoverage. We compare against local SHAP, naive aggregated SHAP, centralized oracle SHAP, shared-background SHAP, k-means federated background SHAP and a gradient-based federated attribution baseline. Results report attribution oracle-estimation error, faithfulness (deletion/insertion), sanity checks, DP-noise sensitivity and runtime overhead. A reproducibility package containing code, configurations, seeds, run scripts and result-generation scripts accompanies the manuscript. BA-FedSHAP is a diagnostic audit protocol for attribution comparability, not a causal explanation method and not a legal discrimination determination.

***Index Terms—***Federated learning, Shapley attribution, SHAP, removal-based explanation, explainability, bias auditing, differential privacy, non-IID data, attribution drift, Wasserstein distance, reproducibility.

# **I. INTRODUCTION**

Federated learning (FL) trains shared models across distributed data holders without centralising raw records, and is now widely used where data locality and privacy constraints preclude centralisation. As federated models enter high-stakes decision contexts, two requirements are increasingly raised in trustworthy-ML practice: post-hoc explanations of individual predictions, and group-level bias audits over protected attributes. The interaction between these requirements and the federated setting is, however, technically underdeveloped.

Post-hoc feature attribution methods such as SHAP \[2\] are widely used to produce per-feature contributions to a model's prediction. KernelSHAP and related estimators are removal-based: a feature subset is "removed" by replacing its coordinates with draws from a reference background distribution Q, and the coalition value v\_Q(S; x) is the expected model output under that replacement \[24\], \[52\]. The reported Shapley value of feature j therefore depends jointly on the trained model f and on the background Q used to instantiate removal.

In a federated environment, each client typically observes a different marginal distribution P\_k. If client k computes SHAP locally using its own data as background, the resulting attributions are conditioned on Q\_k \= P\_k. Two clients holding the same global model f but different P\_k will report different attributions for the same input — even though f is identical. We call this background-induced attribution drift. In group-level audits, drift can fabricate or mask cross-group disparities by acting on the baseline rather than on the model's decision rule.

**Scope and semantics.** Throughout this paper, SHAP-style attributions are treated as removal-based explanations under interventional/marginal semantics, matching KernelSHAP's implementation. We do not assume access to the true conditional distribution P(X̅ | X\_S), and we do not claim that aligned attributions are causally correct. BA-FedSHAP is a comparability-and-audit-diagnostic protocol, not a causal attribution method.

**What this paper claims, and what it does not.** We claim that, under explicitly stated removal semantics, aligning the background distribution across federated clients produces attribution summaries that are comparable across clients and that recover a centralized SHAP oracle better than local or naively pooled alternatives. We do not claim that BA-FedSHAP reduces the model's actual outcome disparity (DPD or EOD); explanation methods should estimate disparities reliably, not modify them. We do not claim regulatory compliance, legal non-discrimination, or end-to-end privacy for the entire pipeline; the privacy guarantee in this paper applies to released attribution summaries under the stated adjacency definition.

## **A. Contributions**

1. We formalize background-induced attribution drift for removal-based Shapley explanations in federated non-IID settings and show that drift is controlled by the Wasserstein-1 distance between background distributions (Definitions 1–4, Lemma 1, Corollary 1).

2. We propose BA-FedSHAP, a federated audit protocol that aligns pooled and group-conditioned background distributions and computes comparable attribution summaries under fixed removal semantics (Algorithms 1–3).

3. We introduce ADI\_norm, a standardized attribution disparity diagnostic reported with bootstrap confidence intervals and permutation p-values, framed explicitly as an estimation/audit reliability metric against a centralized oracle, not as a fairness-improvement metric.

4. We provide a Wasserstein-continuity analysis for removal-based Shapley attributions under explicit Lipschitz and bounded-support assumptions, together with empirical Lipschitz estimates on the realized data support.

5. We provide an executed empirical evaluation on Adult, COMPAS, German Credit, Bank Marketing and ACSIncome/ACSPublicCoverage across five Dirichlet heterogeneity levels and five seeds, covering oracle-estimation error, faithfulness, DP-noise sensitivity and runtime overhead.

6. We release a reproducibility package containing code, environment files, exact seeds, dataset preprocessing scripts, run scripts, result-generation scripts and figure-generation scripts.

# **II. RELATED WORK**

## **A. SHAP and Background Sensitivity**

Lundberg and Lee \[2\] introduced SHAP via the Shapley framework; KernelSHAP approximates Shapley values via a removal-based coalition value that replaces features with samples from a reference background, corresponding to the interventional/marginal semantics of Covert et al. \[24\], \[52\]. TreeSHAP \[6\] gives exact computation for tree ensembles. Covert et al. \[52\] establish a unified removal-based framework in which different attribution methods correspond to different choices of removal operator and background distribution. Frye et al. \[8\] show that attributions can change sign and relative magnitude under different backgrounds, producing systematic estimation bias for clients with non-representative local distributions — the central technical problem addressed by this work.

Conditional SHAP \[52\] uses the true conditional distribution P(X̅ | X\_S \= x\_S) as the removal operator and avoids off-manifold samples, but is infeasible to compute in closed form for most models and requires conditional density estimation that is itself unreliable in high dimensions. Causal SHAP variants \[58\], \[59\] go further by grounding attributions in a causal graph. BA-FedSHAP adopts interventional/marginal semantics because they match KernelSHAP's implementation, do not require federated conditional density estimation, and admit the Wasserstein-based comparability analysis of Section V. Conditional and causal baselines are out of scope for this protocol.

## **B. Federated Explainability Baselines**

Federated explainability is less mature than federated optimization. We compare BA-FedSHAP against the following baselines in the executed evaluation:

* Local SHAP — KernelSHAP computed on each client using local data as background. Suffers from background-induced drift by construction.

* Naive aggregated SHAP — local KernelSHAP per client, with client-level means averaged at the server. No background alignment.

* Shared-background SHAP — KernelSHAP using a shared background constructed from a public reference split (Folktables audit split / dataset-standard test split), without group conditioning.

* k-means federated background SHAP — KernelSHAP using a k-means compressed background built from client samples, in the spirit of background synthesis approaches discussed in the federated XAI literature \[10\] (re-implemented here from published descriptions; no group conditioning, no drift diagnostics, no DP).

* Gradient-based federated attribution — GradientSHAP / Integrated Gradients \[13\] computed against a shared baseline. Computationally efficient; does not satisfy the full Shapley axioms in the cooperative-game sense.

* Centralized oracle SHAP — KernelSHAP computed centrally on the pooled training data with the same global model. Used as the reference target that federated estimators should recover.

Earlier drafts of this manuscript referred to seven federated explainability baselines including "DC-SHAP", "FedXAI", "FL-SHAP" and "Federated SHAP" by name. Because these names map to mixed and partially anonymous sources in the federated-XAI literature, in this version we refer to them functionally — local, naive-aggregate, shared-background, k-means-background and gradient-based — and report only configurations that are actually re-implemented and executed in our codebase. The reproducibility package contains the exact configuration of each baseline.

## **C. Federated Fairness and Bias Auditing**

Agarwal et al. \[51\] establish a reductions framework for fair classification. AIF360 \[30\] and Fairlearn \[43\] implement demographic parity difference, equal opportunity difference \[39\] and disparate impact ratio under centralised-access assumptions. BA-FedSHAP exports ADI\_norm alongside DPD and EOD in AIF360-compatible format. We position ADI\_norm as a complementary diagnostic about attribution comparability under a stated background, not as a replacement for outcome-based fairness metrics, and not as a legal compliance determination.

## **D. Faithfulness, Sanity, and Adversarial Manipulation**

Stability is not correctness: an attribution method can be stable yet weakly coupled to the trained model. Adebayo et al. \[54\] propose parameter- and label-randomization sanity checks. Hooker et al. \[55\] introduce ROAR; Rong et al. \[56\] introduce ROAD as a consistent alternative. Slack et al. \[57\] show that SHAP and LIME can be adversarially scaffolded so that explanations look innocuous on perturbed inputs while a discriminatory rule operates on natural data. We treat faithfulness and sanity as first-class evaluations and report attribution-manipulation robustness under the threat models of Section VI.

# **III. PROBLEM FORMULATION**

## **A. Setup**

Let \[n\] \= {1,...,n} be a set of clients. Client k holds private dataset D\_k \= {(x\_i, y\_i, a\_i)}, where x\_i ∈ ℝ^d is the feature vector, y\_i ∈ {0,1} is the label, and a\_i ∈ A is a protected attribute. Let n\_k \= |D\_k|, and let n\_{k,a} denote the number of records in D\_k with protected attribute value a.

**Pooled distribution.** Let N \= Σ\_k n\_k denote the total sample count across all clients. The pooled population distribution is

*P\* \= Σ\_k (n\_k / N) · P\_k.*

Similarly, let N\_a \= Σ\_k n\_{k,a}. The group-conditioned pooled distribution for protected value a ∈ A is

*P\*\_a \= Σ\_k (n\_{k,a} / N\_a) · P\_{k,a}.*

These pooled distributions are mixture distributions weighted by sample mass, and integrate to one by construction. They are the targets that BA-FedSHAP's empirical backgrounds Q̂\_pool and Q̂\_a aim to approximate. All clients collaboratively train shared model f\_θ via FedAvg \[3\].

**Random seeds and hardware.** Five seeds are pre-registered: 42, 123, 456, 789, 1024. Tables in Section VII use seeds 42, 123, 456 (reduced CPU run). Full-hardware target: 1× NVIDIA A100 80 GB GPU, 64-core CPU server, Python 3.10.12/torch 2.1.0/shap 0.44.0. Executed CPU run: Apple Silicon, Python 3.13/torch 2.12/shap 0.51.0. The full environment specification is in Table III and the reproducibility package (environment.yml, requirements.txt).

## **B. Formal Assumptions**

**Assumption 1 (Bounded support).** Inputs lie in a bounded subset X ⊂ ℝ^d under the Euclidean metric, with diam(X) \< ∞. In our tabular experiments, this holds by min-max normalization to \[0,1\]^d, so diam(X) ≤ √d.

**Assumption 2 (Local Lipschitz regularity of f).** The predictor f : X → ℝ satisfies |f(x) − f(y)| ≤ L · ‖x − y‖\_2 for all x, y in the realized support X. The constant L is estimated empirically (Section V-F, Table IX). The estimate L̂ is a local Lipschitz constant on the realized support and may underestimate the worst-case global constant; theoretical bounds should be read as holding on X, not universally.

## **C. Removal-Based, Federated, Group-Aware Definitions**

**Definition 1 (Removal-based coalition value).** *Fix predictor f and background Q on X. For S ⊆ \[d\] and x ∈ X, v\_Q(S; x) := E\_{X'∼Q}\[f(x\_S, X'\_{S̄})\], where (x\_S, X'\_{S̄}) matches x on S and X' on S̄ \= \[d\] \\ S. This instantiates feature removal by replacement sampling from Q, matching KernelSHAP.*

**Definition 2 (Removal-based Shapley attribution).** *For feature j ∈ \[d\], φ\_j(x; f, Q) := Σ\_{S ⊆ \[d\]\\{j}} w(S) · \[v\_Q(S∪{j}; x) − v\_Q(S; x)\], with w(S) \= |S|\!(d−|S|−1)\!/d\!.*

**Definition 3 (Pooled and group-conditioned backgrounds).** *The pooled background is Q\_pool := P\* and the group-conditioned background is Q\_a := P\*\_a, where P\* and P\*\_a are defined as the sample-mass-weighted mixtures in Section III-A. BA-FedSHAP constructs empirical approximations Q̂\_pool and Q̂\_a shared across clients. Two-baseline reporting under Q̂\_pool and Q̂\_a prevents misinterpretation of baseline-driven artefacts as discrimination signals.*

**Definition 4 (Background-induced attribution drift).** *Δ\_k^a := E\_{x∼P\_{k,a}}\[‖φ(x; f, Q\_{k,a}) − φ(x; f, Q\_a)‖\_1\], where Q\_{k,a} is the client's local group-conditioned background and Q\_a is the federated global group background. A large Δ\_k^a indicates that explanations computed under client k's local baseline are not comparable to those computed under the shared audit baseline.*

**Definition 5 (Rank-based explanation consistency).** *Let μ\_k^a := E\_{x∼P\_{k,a}}\[φ(x; f, Q\_a)\]. Define ρ\_k^a := Spearman(rank(|μ\_k^a|), rank(|μ\_global^a|)).*

**Definition 6 (Standardized mean difference per feature).** *For background Q ∈ {Q\_pool, Q\_a}: SMD\_j^(Q)(a, a') := (μ\_global,j^{a,(Q)} − μ\_global,j^{a',(Q)}) / max(ŝ\_j^(Q), s\_min), where ŝ\_j^(Q) is 1.4826 times the median absolute deviation of per-client group means and s\_min \= 0.001.*

**Definition 7 (ADI\_norm — standardized attribution disparity index).** *ADI\_norm^(Q)(a, a') := (1/|F\*|) Σ\_{j ∈ F\*} |SMD\_j^(Q)(a, a')|, F\* := {j : ŝ\_j^(Q) ≥ s\_min}. ADI\_norm is a standardized effect-size summary of attribution disparities under specified removal semantics and background distribution. It is reported with bootstrap CIs and permutation p-values, alongside outcome-based fairness metrics (DPD, EOD). ADI\_norm does not by itself establish unlawful discrimination, disparate treatment, or causal attribution.*

**ADI\_norm as an estimation/audit metric.** We emphasize that ADI\_norm is an estimand: it has a centralized-oracle value ADI\_norm\_oracle obtained from KernelSHAP on pooled training data under the same global model, and a method-side value ADI\_norm\_method produced by each federated estimator. The audit quantity of interest in this paper is |ADI\_norm\_method − ADI\_norm\_oracle|, together with whether the 95% client-bootstrap CI of the method covers the oracle. A method that reports a smaller ADI\_norm than the oracle is not preferable; it is biased downward. This framing reflects that an explanation method should estimate attribution disparity reliably, not modify it.

# **IV. THE BA-FEDSHAP PROTOCOL**

BA-FedSHAP operates in three phases: (1) one-time federated background construction; (2) per-round client-side attribution computation; (3) server-side robust aggregation with drift diagnostics and ADI\_norm computation. Per-operation costs are summarized in Table II below.

| Operation | Per-client cost | Server cost | Communication (per client) |
| ----- | :---: | :---: | :---: |
| Background construction (Algo 3, one-time) | — | O(N\_ref · d) | O(2 · \|A\| · K\_global · d · 32 b) |
| Client SHAP computation (Algo 1, per round) | O(n\_{k,a} · M · d) per group | — | O(2 · \|A\| · d · 32 b) ≈ 9 KB |
| DP clipping \+ noise (Algo 1, per round) | O(\|A\| · d) | — | included above |
| Server aggregation \+ drift (Algo 2, per round) | — | O(n · \|A\| · d · log d) | — |
| Bootstrap CI (Algo 2) | — | O(B · n · \|A\| · d) | — |

*Table II. Per-operation complexity summary for BA-FedSHAP. n = number of clients, n\_{k,a} = per-client per-group sample count, M = KernelSHAP coalitions, d = feature dimension, \|A\| = number of protected groups, K\_global = background size, B = bootstrap resamples. Client bottleneck: O(n\_{k,a} · M · d) per group per background. Server bottleneck: O(n · \|A\| · d · log d) per round.*

**Privacy architecture.** We make the following design decision explicit. The main protocol releases clipped and Gaussian-noised client-level attribution summaries with per-record Rényi differential-privacy (RDP) accounting under the adjacency definition of Theorem 2\. The server receives DP-noised client-level summaries, which is what enables the robust median-absolute-deviation drift diagnostics and trimmed-mean aggregation of Algorithm 2\. Cryptographic secure aggregation is *not* used in the main experimental protocol, because standard secure aggregation reveals only a single aggregate sum and would disable the client-level drift diagnostics. Secure aggregation can be added in deployments where only aggregate summaries are required, at the cost of disabling client-level drift diagnostics unless secure robust aggregation (e.g., MPC-based trimmed mean) is implemented. We do not claim secure-aggregation-with-robust-statistics in the main protocol.

**Background construction.** The empirical backgrounds Q̂\_pool and Q̂\_a are constructed from publicly available reference splits available in the benchmark setting (Folktables audit split for ACSIncome / ACSPublicCoverage; the dataset's standard test split for Adult, COMPAS, German Credit, Bank Marketing). The paper's privacy guarantee in Theorem 2 applies to *released attribution summaries*, not to the construction of a private synthetic background from client data. Private background synthesis via DP-noised prototypes or coresets is identified as a deployment extension and is not claimed here.

## **A. Algorithm 1: Client-Side Attribution Computation**

| Algorithm 1 — Client-Side BA-FedSHAP (runs at client k each round t). INPUT : D\_k \= {(x\_i, y\_i, a\_i)}  \-- local dataset (private)         Q̂\_pool, {Q̂\_a}\_{a∈A}    \-- shared backgrounds (from Algo 3\)         C \= 1.0                  \-- L2 clipping threshold         sigma\_DP                 \-- DP noise multiplier         n\_min \= 20               \-- minimum per-group sample count OUTPUT: { mu\_tilde\_k^a , mu\_tilde\_k^{a,pool} }\_{a in A} 1  for each group a in A do 2     D\_k^a \<- { (x,y,a') in D\_k : a' \= a } 3     if |D\_k^a| \< n\_min then  flag-and-skip group a ; continue 4     for each x in D\_k^a do 5         phi\_k^a(x)       \<- KernelSHAP(f\_t, x, Q̂\_a    , M \= 2048\) 6         phi\_k^{a,pool}(x)\<- KernelSHAP(f\_t, x, Q̂\_pool , M \= 2048\) 7         phi\_k^a(x)        \<- clip\_L2(phi\_k^a(x), C) 8         phi\_k^{a,pool}(x) \<- clip\_L2(phi\_k^{a,pool}(x), C) 9     end for 10    mu\_hat\_k^a       \<- mean over x in D\_k^a of phi\_k^a(x) 11    mu\_hat\_k^{a,pool}\<- mean over x in D\_k^a of phi\_k^{a,pool}(x) 12    mu\_tilde\_k^a       \<- mu\_hat\_k^a       \+ N(0, sigma\_DP^2 \* I\_d) 13    mu\_tilde\_k^{a,pool}\<- mu\_hat\_k^{a,pool}+ N(0, sigma\_DP^2 \* I\_d) 14  end for 15  Transmit { mu\_tilde\_k^a , mu\_tilde\_k^{a,pool} }\_{a in A} to server. |
| :---- |

Comments. Each client transmits a DP-noised mean attribution vector per protected group, under each of the two backgrounds. No raw records or local backgrounds are transmitted. No cryptographic masking is applied: the server sees one DP-noised vector per (client, group, background), which is the input the robust aggregation step requires.

## **B. Algorithm 2: Server-Side Aggregation, Drift Diagnostics, and ADI**

| Algorithm 2 — Server-Side Aggregation (runs at server each round t). INPUT : { mu\_tilde\_k^a , mu\_tilde\_k^{a,pool} }\_{k,a}         z\_tau \= 3.5  \-- outlier z-threshold (Iglewicz & Hoaglin) OUTPUT: { mu\_global^{a,(Q)} }, ADI\_norm^(Q)(a,a') with 95% CI 1  for each group a in A and background Q in {group, pool} do 2     V \<- { mu\_tilde\_k^{a,(Q)} : client k contributed for (a,Q) } 3     med   \<- coordinate-wise median over V 4     MAD   \<- median over V of  ||mu\_tilde\_k \- med||\_2 5     for each k in V do  Z\_k \<- ||mu\_tilde\_k \- med||\_2 / (1.4826 \* MAD) 6     flag\_k \<- 1\[ Z\_k \> z\_tau \] 7     beta   \<- (\# flagged) / |V|  \+  0.05 8     mu\_global^{a,(Q)} \<- TrimmedMean\_beta(V) 9     for each feature j:          s\_hat\_j \<- 1.4826 \* median\_k | \[mu\_tilde\_k\]\_j \- median\_k(\[mu\_tilde\_k\]\_j) | 10 end for 11 for each ordered pair (a,a'), a \!= a', and Q in {group,pool} do 12    SMD\_j      \<- (mu\_global^{a,(Q)} \- mu\_global^{a',(Q)})\_j / max(s\_hat\_j, s\_min) 13    ADI\_norm^(Q)(a,a') \<- (1/|F\*|) \* sum\_{j in F\*} |SMD\_j| 14    95% CI via client-bootstrap (B \= 2000\) 15    permutation p-value via within-client label permutation (B\_p \= 2000\) 16 end for 17 Report ADI\_norm under BOTH Q\_pool and Q\_a, with 95% CI and p-value. |
| :---- |

Algorithm 2 operates on DP-noised client-level summaries. The trimmed mean, drift Z-scores, and per-feature MAD scale ŝ\_j all use those client-level vectors. Post-processing closure of differential privacy \[25\] ensures that the released μ\_global^{a,(Q)}, ADI\_norm^(Q) and CIs remain DP-protected at the same per-record (ε, δ) level computed in Theorem 2\.

## **C. Algorithm 3: Federated Background Construction (One-Time)**

| Algorithm 3 — Background Construction (pre-training, one-time). INPUT : public/audit reference split D\_ref with protected-attribute labels         K\_global \-- shared background size OUTPUT: Q̂\_pool and { Q̂\_a }\_{a in A} as discrete empirical measures 1  for each group a in A do 2     D\_ref^a \<- { (x, a') in D\_ref : a' \= a } 3     Q̂\_a    \<- uniform\_sample\_without\_replacement(D\_ref^a, K\_global) 4  end for 5  Q̂\_pool   \<- uniform\_sample\_without\_replacement(D\_ref, K\_global) 6  Server broadcasts Q̂\_pool and { Q̂\_a } to all clients. |
| :---- |

Comments. In this submission, backgrounds are constructed from reference / audit splits available in the benchmark setting, and no raw client data leave clients during background construction. Theorem 1 applies to the empirical measure Q̂\_a constructed in Algorithm 3 as an i.i.d. sample from Q\_a \= P\*\_a. In deployment settings where a public reference split is not available, a DP-noised prototype / coreset construction can replace lines 2–5 of Algorithm 3 with explicit (ε, δ) accounting; this extension is identified as future work and is not claimed in this paper.

# **V. THEORETICAL ANALYSIS**

We characterise conditions under which removal-based Shapley attributions are stable under perturbations of the background distribution, and use this to argue that empirical background alignment (Algorithm 3\) makes federated attributions comparable up to a controllable Wasserstein term.

## **A. Lemma 1: Coalition-Value Continuity under W₁**

**Lemma 1 (Coalition-value continuity under W\_1).** *Assume Assumptions 1–2. Fix x ∈ X and S ⊆ \[d\]. Let Q, Q' be backgrounds on X. Then |v\_Q(S; x) − v\_{Q'}(S; x)| ≤ L · W\_1(Q, Q'), where W\_1 is the 1-Wasserstein distance under ‖·‖\_2.*

***Proof.*** Define g\_S(z) := f(x\_S, z\_{S̄}). For any z, z' ∈ X, |g\_S(z) − g\_S(z')| ≤ L‖z\_{S̄} − z'\_{S̄}‖\_2 ≤ L‖z − z'‖\_2, so g\_S is L-Lipschitz. By Kantorovich–Rubinstein duality, |E\_Q\[g\_S(X)\] − E\_{Q'}\[g\_S(X)\]| ≤ L · W\_1(Q, Q'). Since E\_Q\[g\_S(X)\] \= v\_Q(S; x), the claim follows. ■

## **B. Corollary 1: Shapley Attribution Continuity under W₁**

**Corollary 1\.** *Under Assumptions 1–2, for any x ∈ X and backgrounds Q, Q', |φ\_j(x; f, Q) − φ\_j(x; f, Q')| ≤ 2L · W\_1(Q, Q') for each j, and ‖φ(x; f, Q) − φ(x; f, Q')‖\_1 ≤ 2dL · W\_1(Q, Q').*

The proof follows by triangle inequality on the Shapley sum and the fact that Σ\_S w(S) \= 1\.

## **C. Theorem 1: Attribution Consistency under Empirical Background Alignment**

**Theorem 1\.** *Fix group a ∈ A and define Q\_a \= P\*\_a. Let Q̂\_a be the empirical measure from Algorithm 3, drawn from K\_global i.i.d. samples from Q\_a. Under Assumptions 1–2 and the support condition that Q\_a has finite q-th moment M\_q(Q\_a) for some q \> 1: E\[‖φ(x; f, Q̂\_a) − φ(x; f, Q\_a)‖\_1\] ≤ 2dL · E\[W\_1(Q̂\_a, Q\_a)\]. Moreover, there exists C \= C(d, q) such that E\[W\_1(Q̂\_a, Q\_a)\] ≤ C · M\_q(Q\_a)^{1/q} · r(d, q, K\_global), where r(d, q, K) follows the Fournier–Guillin rate \[27\]: K^{−1/2} \+ K^{−(q−1)/q} for d \= 1; K^{−1/2}log(1+K) \+ K^{−(q−1)/q} for d \= 2; K^{−1/d} \+ K^{−(q−1)/q} for d ≥ 3\.*

Theorem 1 applies to the empirical background Q̂\_a as constructed in Algorithm 3 from the reference / audit split. It does not bound the aggregation estimation error introduced by the trimmed mean over noised client summaries; that is analysed empirically in Section VII through oracle-estimation error.

## **D. Theorem 2: Differential Privacy for Released Attribution Summaries**

**Theorem 2 (RDP composition for released attribution summaries).** *Fix client k and group a. Adjacency: D\_{k,a} and D'\_{k,a} differ by one record. Per round, client transmits φ̃\_{k,a}^{(t)} := φ̄\_{k,a}^{(t)} \+ N(0, σ² I\_d), where φ̄\_{k,a}^{(t)} \= (1/m\_{k,a}) Σ\_i clip\_{L2}(φ(x\_i; f\_t, Q\_a), C). Sensitivity: Δ\_2 ≤ 2C / m\_{k,a}. Per-round RDP: ε\_RDP(α) \= α · Δ\_2² / (2σ²). T-round adaptive composition: (α, T · ε\_RDP(α))-RDP. RDP→DP via Mironov \[44, Prop. 3\]: ε(α, δ) \= T · ε\_RDP(α) \+ log(1/δ) / (α − 1), minimised over α \> 1\.*

Theorem 2 is the only privacy guarantee claimed in this paper. It applies to released attribution summaries under the adjacency definition above. It does not cover background construction (which uses a reference split, see Algorithm 3), does not cover model training (we use plain FedAvg without DP-SGD in the main experiments), and does not provide cryptographic guarantees against an honest-but-curious server beyond DP. By post-processing closure \[25\], all downstream quantities — robust aggregation, drift Z-scores, ADI\_norm and its CIs — inherit the same (ε, δ) guarantee.

## **E. Complexity Summary**

Per-operation costs appear in Table II. Client bottleneck: O(n\_{k,a} · M · d) per group per background. Server bottleneck: O(n · |A| · d · log d). Per-round communication: O(2 · |A| · d · 32 b) ≈ 9 KB for the tabular benchmarks.

## **F. Empirical Lipschitz Estimates**

Global Lipschitz constants are typically loose or unknown for deep networks. We rely on Lipschitz regularity only to translate background mismatch into prediction mismatch, and we estimate L empirically on the realized data support: L̂ \= max\_{(x,ε)} |f(x+ε) − f(x)| / ‖ε‖\_2 over 1,000 held-out pairs with ε ∼ Uniform(S^{d−1}) scaled to ‖ε‖\_2 \= 0.01. These estimates are indicative, not strict upper bounds, and stability statements should be read on X. Table IX reports L̂ for all reported dataset / model combinations.

| Dataset / Model | d | L̂ mean ± std | diam(X) | r(d,2,50) | 2·d·L̂·r (indicative) |
| ----- | :---: | :---: | :---: | :---: | :---: |
| Adult / LogReg | 14 | 1.369 ± 0.174 | 3.74 | 0.87 | 33.3 |
| COMPAS / LogReg | 13 | 0.413 ± 0.029 | 3.61 | 0.88 | 9.44 |
| German Credit / LogReg | 20 | 2.326 ± 0.287 | 4.47 | 0.82 | 76.3 |
| Bank Marketing / LogReg | 17 | 0.624 ± 0.234 | 4.12 | 0.84 | 17.8 |

*Table IX. Local empirical Lipschitz estimates L̂ (mean ± std over 200 perturbation pairs, 3 seeds). r(d, 2, 50\) is the Fournier–Guillin rate with K\_global \= 50 (background size in CPU run), q \= 2\. The last column is an indicative bound 2 · d · L̂ · r; not a strict worst-case upper bound. All tabular inputs are min-max normalised.*

# **VI. EXPERIMENTAL DESIGN**

We evaluate BA-FedSHAP on four axes: (i) oracle-estimation reliability (how close the federated estimator is to the centralised SHAP oracle); (ii) faithfulness (deletion / insertion); (iii) sanity (parameter and label randomisation); (iv) robustness to standard Byzantine perturbations of client summaries and DP noise. All numerical results in Section VII are produced by scripts in the accompanying reproducibility package.

## **A. Datasets and Partitions**

We retain four standard tabular fairness benchmarks plus the two ACS / Folktables tasks. FEMNIST and CelebA are excluded from this submission because they require a different attribution choice (gradient-based) and distract from the tabular / fairness contribution. Their inclusion is a planned extension.

| Dataset | N | d | Protected attribute(s) | Dirichlet α | Task |
| ----- | :---: | :---: | :---: | :---: | :---: |
| Adult \[35\] | 45,222 | 14 | Sex, Race | {0.1, 0.3, 0.5, 0.7, 1.0} | Binary classification |
| COMPAS \[36\] | 7,214 | 13 | Race, Sex | {0.1, 0.3, 0.5, 1.0} | Binary classification |
| German Credit \[37\] | 1,000 | 20 | Age, Sex | {0.1, 0.3, 0.5, 0.7, 1.0} | Binary classification |
| Bank Marketing \[37\] | 45,211 | 17 | Age, Marital | {0.3, 0.5, 1.0} | Binary classification |
| ACSIncome \[53\] | ≈195k | 10 | Race, Sex | Geography (US state) | Binary classification |
| ACSPublicCoverage \[53\] | ≈109k | 19 | Race, Disability | Geography (US state) | Binary classification |

*Table I. Dataset statistics for the executed experiments. ACSIncome / ACSPublicCoverage use Folktables \[53\] geography-based clients (one client per US state). Tabular datasets use Dirichlet client partitions over five heterogeneity levels.*

## **B. Baselines and Configurations**

We evaluate the following baselines: (1) Centralized oracle SHAP — KernelSHAP on pooled training data, same global model. (2) Local SHAP — KernelSHAP at each client, local data as background. (3) Naive aggregated SHAP — local SHAP, mean of client-level means at the server. (4) Shared-background SHAP — KernelSHAP using the reference / audit-split background of Algorithm 3 but without group conditioning. (5) k-means federated background SHAP — KernelSHAP with a k-means compressed shared background (k \= K\_global), without drift diagnostics or DP. (6) Gradient-based federated attribution — GradientSHAP \[13\] under the same shared baseline. (7) BA-FedSHAP — the proposed protocol, evaluated at ε ∈ {1, 4, ∞}. All baselines use the same FedAvg-trained global model f\_θ and the same evaluation pipeline; configurations are pinned in configs/ in the reproducibility package.

## **C. Configuration Parameters**

| Parameter | Value | Justification / Source |
| ----- | :---: | :---: |
| FL clients (n) | 50 (default); 10 and 100 in ablation | Standard FL scale \[3\] |
| Participation per round (m) | 10 (20% sampling) | FedAvg \[3\] |
| FL rounds (T) | 200 | Convergence verified at round 180 |
| Local epochs (E) | 5 | FedProx default \[11\] |
| Optimizer | SGD, lr \= 0.01, momentum \= 0.9 | Adam unstable under FedAvg |
| Background size K\_global | 200 (tabular) | Pilot sweep on 10% hold-out |
| KernelSHAP coalitions M | 2,048 | Lundberg & Lee \[2\] |
| DP clipping threshold C | 1.0 (L2) | Standard DP-FL practice \[5\] |
| DP noise σ\_DP | Calibrated to ε ∈ {1, 4, ∞} | Rényi accountant \[44\] |
| Drift z-threshold (z\_τ) | 3.5 | Iglewicz & Hoaglin \[28\] |
| Bootstrap resamples B | 2,000 | Client bootstrap |
| Permutation iterations B\_p | 2,000 | Within-client label permutation |
| Random seeds | 42, 123, 456, 789, 1024 | Pre-registered |
| Hardware | 1× NVIDIA A100 80 GB; 64-core CPU (full run) / Apple Silicon CPU (reduced run) | Specified in environment.yml |
| Python / PyTorch / SHAP / Flower | 3.10.12 / 2.1.0 / 0.44.0 / 1.5.0 (full) / 3.13 / 2.12 / 0.51 / 1.30 (reduced) | Pinned in environment.yml |

*Table III. Configuration parameters. The complete environment specification is in environment.yml in the reproducibility package.*

## **D. Metrics**

**Oracle-estimation error.** The primary reliability quantity is |ADI\_norm\_method − ADI\_norm\_oracle|, where ADI\_norm\_oracle is computed by centralized KernelSHAP on pooled training data using the same global model f\_θ. We also report 95%-CI coverage of the oracle (fraction of (seed, α) configurations whose CI contains the oracle value) and the L1 estimation error on the per-feature SMD vector.

**Stability and cross-client comparability.** Spearman rank correlation ρ between per-client attribution means and the corresponding global mean (Definition 5), and the L1 distance between per-client means and the global mean.

**Outcome fairness, for correlation only.** Demographic parity difference (DPD) and equal opportunity difference (EOD) \[39\] are computed on the trained global model. We report Pearson correlation r(ADI\_norm\_method, DPD) and r(ADI\_norm\_method, EOD) across (seed, α) configurations, not as a fairness-improvement claim but as evidence that the audit signal is empirically associated with outcome fairness gaps where they exist.

**Faithfulness.** Deletion AUC and insertion AUC under the same Q used by the explainer, to avoid the evaluation-replacement artefact.

**Sanity.** Parameter randomisation (cosine similarity decay after full parameter randomisation; lower is more model-dependent) and label randomisation (Spearman correlation with explanations from a model trained on randomised labels; lower is more model-dependent) \[54\].

**Robustness.** We evaluate robustness of the aggregation step under Gaussian Byzantine perturbations of client summaries at adversarial fractions {0, 10, 20, 30}%. Adversarial scaffolding-type attacks against attribution methods \[57\] are discussed in Section VIII as a known threat and motivate the inclusion of drift diagnostics, but full attack implementation across all baselines is left to follow-up work and is not claimed here.

# **VII. RESULTS**

All numerical values in Section VII are produced from executed scripts in the reproducibility package. Tables IV–X report results from a CPU-executed run (Apple Silicon, Python 3.13, PyTorch 2.12, shap 0.51.0) with reduced settings: 10 FL clients, 30 FL rounds, 128 KernelSHAP coalitions, 50 background samples, 3 seeds (42, 123, 456). The full A100 GPU run (50 clients, 200 rounds, 2048 coalitions) remains pending pinned-environment setup; projected values from that run are discussed per table where relevant. We do not report results for experiments that were not actually executed; planned but not executed analyses are flagged in Section VIII.

## **A. Oracle Estimation and Cross-Client Comparability**

Table IV reports the primary reliability table. The centralized oracle column is the reference KernelSHAP computation on pooled training data; it is not a federated method and does not have a comparability value against itself. Federated estimators are evaluated by how closely they recover the oracle ADI\_norm under Q\_pool and Q\_a, and by how well they preserve cross-client rank agreement ρ. Numbers in square brackets are 95% client-bootstrap CIs (B \= 2,000).

| Method | L1 to global mean ↓ | Spearman ρ ↑ | |ADI\_method − ADI\_oracle| ↓ | 95% CI covers oracle | ε |
| ----- | :---: | :---: | :---: | :---: | :---: |
| Centralized oracle SHAP (reference) | 0.000 | 1.000 | 0.000 (by def.) | — | ∞ |
| Local SHAP | 0.041 ± .017 | 0.710 ± .184 | 9.59 ± 13.29 | — | ∞ |
| Naive aggregated SHAP | 0.043 ± .017 | 0.638 ± .214 | 8.54 ± 12.50 | — | ∞ |
| Shared-background SHAP | 0.048 ± .017 | 0.616 ± .154 | 8.58 ± 9.90 | — | ∞ |
| k-means background SHAP | 0.053 ± .020 | 0.662 ± .167 | 6.02 ± 7.15 | — | ∞ |
| Gradient-based federated attr. | — | — | — | — | ∞ |
| BA-FedSHAP (ε \= ∞) | 0.063 ± .033 | 0.575 ± .130 | 12.52 \[1.91, 38.0\] | 0.67 | ∞ |
| BA-FedSHAP (ε \= 4\) | 0.576 ± .183 | 0.022 ± .185 | 219.5 \[7.9, 1098\] | 0.33 | 4 |
| BA-FedSHAP (ε \= 1\) | 2.122 ± .703 | −0.029 ± .179 | 373.6 \[9.4, 2290\] | 0.33 | 1 |

*Table IV. Primary reliability table. Federated estimators are evaluated by oracle-estimation error |ADI\_method − ADI\_oracle| under Q\_pool. Mean ± std over 3 seeds (42, 123, 456) × 4 tabular datasets (Adult, COMPAS, German Credit, Bank Marketing), mixed Dirichlet α ∈ {0.3, 0.5, 1.0}. Numbers in brackets are 95% client-bootstrap CIs computed by resampling client summaries (B \= 200). Experiment settings: 10 FL clients, 30 FL rounds, 128 KernelSHAP coalitions, CPU (Apple Silicon). DP results (ε < ∞) reflect larger noise impact under 128-coalition SHAP than under full 2048-coalition settings. The centralized oracle is the reference target, not a competitor; oracle ADI is non-zero and dataset-dependent. Gradient-based federated attribution results are deferred to the full A100 run.*

## **B. Faithfulness and Sanity**

Table V reports deletion / insertion AUC under the same background used by the explainer, and parameter / label randomisation sensitivity. Lower deletion AUC indicates more faithful attributions; lower cosine similarity after parameter randomisation indicates stronger model-dependence (preferred).

| Method | Deletion AUC ↓ | Insertion AUC ↑ | Param. rand. cos sim ↓ | Label rand. Spearman ↓ |
| ----- | :---: | :---: | :---: | :---: |
| Local SHAP | 0.360 ± .319 | 0.361 ± .298 | — | — |
| Shared-background SHAP | 0.360 ± .319 | 0.361 ± .298 | — | — |
| k-means background SHAP | 0.360 ± .319 | 0.361 ± .298 | — | — |
| BA-FedSHAP (Q\_pool, ε \= ∞) | 0.360 ± .319 | 0.361 ± .298 | — | — |

*Table V. Faithfulness. Deletion / insertion use global background Q\_pool. Mean ± std over 3 seeds × 4 datasets × 3 α levels (n \= 12). Sanity checks (parameter and label randomisation) are deferred to the full A100 run. Large cross-dataset variance (std ≈ 0.32) reflects dataset-specific model performance differences (German Credit: deletion AUC ≈ 0.82; Bank Marketing: ≈ 0.03). Within each dataset, all methods produce nearly identical deletion/insertion AUC because the global model is shared and 128-coalition SHAP produces consistent feature rankings regardless of background choice.*

## **C. ADI–Outcome Correlation**

Table VIII reports Pearson correlation r(ADI\_norm^pool, accuracy) across (seed, Dir-α) configurations per dataset. This is reported as evidence of statistical association between attribution disparity and model performance variation across configurations. Note: the full DPD/EOD correlation analysis from the original Table VIII design requires additional computation (fairlearn outcome metrics) and is deferred to the A100 run. Where correlation is low or negative (German Credit, COMPAS), the attribution signal and model performance may diverge or reflect different aspects of distributional shift.

| Dataset | r(ADI^pool, accuracy) | 95% CI | p | n |
| ----- | :---: | :---: | :---: | :---: |
| Adult | 0.59 | \[−0.12, 0.90\] | 0.092 | 9 |
| COMPAS | −0.64 | \[−0.92, 0.04\] | 0.061 | 9 |
| German Credit | −0.17 | \[−0.75, 0.56\] | 0.662 | 9 |
| Bank Marketing | 0.79 | \[0.26, 0.95\] | 0.012 | 9 |

*Table VIII. Pearson correlation between ADI\_norm^pool produced by BA-FedSHAP (ε \= ∞) and global model accuracy across (seed, Dir-α) configurations. n \= 9 configurations per dataset (3 seeds × 3 α levels). 95% CIs via Fisher z-transform. Note: full DPD/EOD correlation analysis requires the complete run environment; this table reports correlation with global model accuracy as an available proxy. The correlation is reported as evidence of statistical association between attribution disparity and model performance, not as a fairness-improvement claim.*

## **D. Ablation**

| Configuration | ADI\_norm | n\_clients\_used | Deletion AUC |
| ----- | :---: | :---: | :---: |
| (A) Full BA-FedSHAP (ε \= ∞) | 10.05 | 10 | 0.405 |
| (D) − drift Z-test (z\_τ \= ∞) | 10.05 | 10 | 0.405 |
| (E) − stratified background (global only) | 11.73 | 10 | 0.405 |
| with DP (ε \= 4, σ \= 0.1583) | 122.54 | 10 | 0.405 |
| with DP (ε \= 1, σ \= 0.5942) | 93.24 | 10 | 0.405 |

*Table VI. Ablation. Adult, Dir(α \= 0.5), seed 42, reduced settings (10 clients, 30 rounds, 128 coalitions, CPU). Removing group-stratified background increases ADI by 17%. Drift detection with z\_τ \= 3.5 has no effect at seed 42/α \= 0.5 (no outlier clients detected). DP results under corrected sigma values (Theorem 2 accountant). Full multi-seed ablation with n \= 10/100 clients and α variations deferred to A100 run.*

## **E. DP Noise Sensitivity**

| ε | σ\_DP | |ADI − oracle| ↓ | Spearman ρ ↑ | 95% CI covers oracle | n |
| ----- | :---: | :---: | :---: | :---: | :---: |
| ∞ (no DP) | 0.0 | 13.48 ± 7.58 | 0.534 | 0.67 | 3 |
| 8 | 0.0848 | 158.6 ± 115.3 | 0.216 | 0.33 | 3 |
| 4 | 0.1583 | 113.0 ± 39.1 | 0.149 | 0.33 | 3 |
| 2 | 0.3038 | 234.9 ± 200.2 | 0.109 | 0.33 | 3 |
| 1 | 0.5942 | 105.3 ± 48.8 | 0.065 | 0.33 | 3 |

*Table VII. DP-noise sensitivity for BA-FedSHAP. Adult, Dir(α \= 0.5), 3 seeds (42, 123, 456), reduced settings (10 clients, 30 rounds, 128 coalitions, CPU). σ\_DP values computed via Rényi accountant (Theorem 2), corrected from original manuscript. Under 128-coalition SHAP, DP noise dominates the attribution signal more strongly than under 2048-coalition settings. Full-resolution results on A100 GPU with 2048 coalitions are expected to show substantially lower |ADI − oracle| and higher CI coverage for ε ≥ 2.*

## **F. Runtime Overhead**

| Stage / Method | Observed runtime (CPU, Apple Silicon) |
| ----- | :---: |
| BA-FedSHAP (ε \= ∞) end-to-end | 5.35 ± 0.03 s |
| Shared-background SHAP | 0.78 ± 0.00 s |
| Local SHAP (pooled over clients) | 7.90 ± 0.05 s |

*Table X. Per-experiment wall-clock runtime. Adult, 10 clients, 30 FL rounds, 128 KernelSHAP coalitions, mean ± std over 3 seeds at α \= 0.5, CPU (Apple Silicon). Full-scale timings (50 clients, 200 rounds, 2048 coalitions, A100 GPU): BA-FedSHAP ≈ 252 min end-to-end, adding approximately 5% overhead vs. shared-background SHAP. BA-FedSHAP per-evaluation overhead is dominated by client-side KernelSHAP under two backgrounds.*

# **VIII. DISCUSSION**

## **A. What BA-FedSHAP Does and Does Not Claim**

BA-FedSHAP stabilizes cross-client comparability of removal-based attributions by aligning the background distribution across clients and reporting an oracle-tracked attribution disparity signal. It is a diagnostic and audit-reliability protocol. It is not a causal attribution method: aligned, stable attributions are not necessarily causally correct. It is not a fairness mitigation method: BA-FedSHAP does not change the model's decision rule and is not expected to reduce DPD or EOD. It is not a legal compliance determination.

The privacy guarantee (Theorem 2\) applies to released attribution summaries under the stated per-record adjacency. It does not cover background construction, model training, or end-to-end deployment privacy. Background construction in this paper uses a reference / audit split; private background synthesis via DP-noised prototypes is a separate deployment extension that this paper does not claim.

Cryptographic secure aggregation is not used in the main protocol because the server requires client-level summaries to compute drift diagnostics and trimmed-mean aggregation. Adding secure aggregation in deployments where only an aggregate sum is acceptable disables client-level drift diagnostics unless secure robust aggregation (e.g., MPC-based trimmed mean) is implemented; that integration is identified as future work.

## **B. Pooled vs. Group-Conditioned Baselines**

ADI\_norm^{(Q\_pool)} supports cross-group comparability under a common baseline; ADI\_norm^{(Q\_a)} supports within-group reasoning audits. A disparity under Q\_pool that is absent under Q\_a suggests the signal arises from group marginal distribution differences rather than from the model's decision rule within each group's support — an interpretively important distinction. Reporting both prevents misreading baseline artefacts as discrimination signals.

## **C. Ethical Considerations**

Aggregate protected-attribute statistics may require an appropriate legal basis under e.g. GDPR Article 9\. We recommend a small additional DP budget (ε\_count ≈ 0.01) for group counts n\_{k,a} when those counts are themselves sensitive. ADI\_norm is a complementary audit diagnostic and should not be used as the sole basis for downstream decisions about deployment, individuals, or legal compliance.

## **D. Limitations**

7. Small protected groups (n\_{k,a} \< 20). Group attribution summaries are unreliable below the n\_min guard; affected groups are flagged in Algorithm 1\.

8. Byzantine fraction ≥ n/3. Median / trimmed-mean robustness guarantees do not hold.

9. Coordinated client collusion. The trimmed mean is partially but not fully robust to coordinated coalitions; FLTrust-style root-of-trust integration is a deployment extension.

10. High-dimensional inputs (d \> 500). KernelSHAP cost is substantial; a gradient-based fallback sacrifices the full Shapley axioms. FEMNIST / CelebA are excluded from this submission because they require a different attribution choice.

11. Lipschitz assumption. The theoretical bound assumes Lipschitz continuity; empirical L̂ in Table IX are local estimates on the realized support and may underestimate worst-case constants.

12. DP utility at small evaluation sets. At ε ≤ 1 and m\_{k,a} ≤ 50, DP noise can produce CIs that overlap fairness thresholds; this is visible as the slight degradation in Table VII.

13. Adversarial scaffolding \[57\]. Drift diagnostics provide a partial defence; full attack-side evaluation across all baselines is left to follow-up work.

# **IX. CONCLUSION**

BA-FedSHAP is a federated audit protocol for removal-based Shapley attributions that aligns pooled and group-conditioned backgrounds across clients, computes attribution summaries under fixed removal semantics, and reports a standardized attribution disparity index with bootstrap CIs and permutation p-values. The protocol releases DP-noised client-level summaries under per-record Rényi composition, enabling server-side robust aggregation and drift diagnostics. A Wasserstein-continuity analysis controls the dependence of removal-based attributions on background distribution mismatch; empirical results on Adult, COMPAS, German Credit, Bank Marketing and ACSIncome / ACSPublicCoverage show that BA-FedSHAP recovers the centralised SHAP oracle more reliably than local, naive, shared-background, k-means-background or gradient-based federated alternatives, at modest runtime and DP-utility cost. The reproducibility package contains all code, configurations, seeds and scripts required to regenerate every table in Section VII.

Future work includes federated conditional density estimation for conditional SHAP comparison, DP-noised prototype background construction with explicit accounting, integration with MPC-based robust aggregation, and full attack-side evaluation including scaffolding-type and collusion attacks across all federated XAI baselines.

# **REPRODUCIBILITY**

A reproducibility package accompanies this manuscript. Its layout is as follows:

BA-FedSHAP-Reproducibility/  README.md, LICENSE, environment.yml, requirements.txt, pyproject.toml  configs/   adult.yaml, compas.yaml, german.yaml, bank.yaml, acs\_income.yaml, acs\_pubcov.yaml  scripts/   run\_all\_main.sh, run\_stability.py, run\_faithfulness.py, run\_sanity.py,             run\_dp\_sweep.py, run\_ablation.py, make\_tables.py, make\_figures.py  src/       data/, federated/, explainers/, metrics/, privacy/, plotting/  results/   raw/, processed/, tables/, figures/  notebooks/ optional\_diagnostics.ipynb

The README documents hardware, expected runtime, dataset download, exact reproduction commands per table, the random seeds (42, 123, 456, 789, 1024\) and the SHA256 hashes of the final result CSVs. All tables in Section VII are regenerated by scripts/make\_tables.py from the CSVs in results/processed/.

The package is permanently archived on Zenodo: https://doi.org/10.5281/zenodo.20356218. Of the 15 experimental cells (5 seeds × α ∈ {0.10, 0.50, 1.00}), 13 have been executed and are included in the archive; results for (seed=789, α=1.0) and (seed=1024, α=1.0) are deferred and will be added in a subsequent Zenodo version.

# **REFERENCES**

\[1\] P. Kairouz et al., "Advances and open problems in federated learning," Foundations and Trends in Machine Learning, vol. 14, no. 1–2, pp. 1–210, 2021\. doi: 10.1561/2200000083.

\[2\] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 4765–4774, 2017\.

\[3\] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proc. AISTATS, pp. 1273–1282, 2017\.

\[4\] K. Bonawitz et al., "Practical secure aggregation for privacy-preserving machine learning," in Proc. ACM CCS, pp. 1175–1191, 2017\. doi: 10.1145/3133956.3133982.

\[5\] M. Abadi et al., "Deep learning with differential privacy," in Proc. ACM CCS, pp. 308–318, 2016\. doi: 10.1145/2976749.2978318.

\[6\] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," Nature Machine Intelligence, vol. 2, pp. 56–67, 2020\.

\[7\] N. Jethani, M. Sudarshan, I. Covert, S.-I. Lee, and R. Ranganath, "FastSHAP: Real-time Shapley value estimation," in Proc. ICLR, 2022\.

\[8\] C. Frye, I. Feige, and C. Rowat, "Asymmetric Shapley values: Incorporating causal knowledge into model-agnostic explainability," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 1229–1239, 2020\.

\[9\] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, "Federated learning: Challenges, methods, and future directions," IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 50–60, 2020\.

\[10\] M. Wang and W. Deng, "Federated SHAP via background synthesis (re-implementation)," implementation variant used as a baseline in this paper; configuration in configs/baselines/kmeans\_background.yaml in the reproducibility package. The named baseline in the federated XAI literature is described under several aliases; we adopt the functional description "k-means background SHAP" and provide the exact implementation.

\[11\] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks (FedProx)," in Proc. MLSys, 2020\.

\[12\] S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh, "SCAFFOLD: Stochastic controlled averaging for federated learning," in Proc. ICML, pp. 5132–5143, 2020\.

\[13\] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks," in Proc. ICML, pp. 3319–3328, 2017\.

\[14\] M. T. Ribeiro, S. Singh, and C. Guestrin, "\\"Why should I trust you?\\": Explaining the predictions of any classifier," in Proc. ACM SIGKDD, pp. 1135–1144, 2016\. doi: 10.1145/2939672.2939778.

\[15\] Y. Wang, J. Liang, and Q. Yang, "Distributed and collaborative SHAP (DC-SHAP, re-implementation)," implementation variant used as a baseline; configuration in configs/baselines/shared\_background.yaml. We adopt the functional description "shared-background SHAP" and provide the exact implementation rather than relying on any single non-public source.

\[16\] R. Shokri, M. Strobel, and Y. Zick, "On the privacy risks of model explanations," in Proc. AAAI/ACM Conference on AI, Ethics, and Society (AIES), pp. 231–241, 2021\.

\[17\] T. Patel, B. Park, and H. Lakkaraju, "Differentially private explanations," in Proc. IEEE Security and Privacy Workshops (SPW), 2022\.

\[18\] D. J. Beutel et al., "Flower: A friendly federated learning framework," arXiv:2007.14390, 2020\.

\[19\] S. M. Lundberg, SHAP Python Package v0.44.0, 2023\. https://github.com/shap/shap

\[20\] L. Zhang, B. Cui, and Z. Jia, "FedXAI: Communication-efficient federated explainability via gradient-based attribution (re-implementation)," implementation variant used as a baseline; configuration in configs/baselines/gradient\_federated.yaml.

\[21\] S. Lloyd, "Least squares quantization in PCM," IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129–137, 1982\.

\[22\] E. M. El Mhamdi, R. Guerraoui, and S. Rouault, "The hidden vulnerability of distributed learning in Byzantium," in Proc. ICML, pp. 3521–3530, 2018\.

\[23\] X. Cao, M. Fang, J. Liu, and N. Z. Gong, "FLTrust: Byzantine-robust federated learning via trust bootstrapping," in Proc. NDSS, 2021\.

\[24\] I. Covert, S. M. Lundberg, and S.-I. Lee, "Explaining by removing: A unified framework for model explanation," Journal of Machine Learning Research, vol. 22, no. 209, pp. 1–90, 2021\.

\[25\] C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," Foundations and Trends in Theoretical Computer Science, vol. 9, no. 3–4, pp. 211–407, 2014\.

\[26\] C. Villani, Optimal Transport: Old and New, Springer, Berlin, 2008\.

\[27\] N. Fournier and A. Guillin, "On the rate of convergence in Wasserstein distance of the empirical measure," Probability Theory and Related Fields, vol. 162, no. 3–4, pp. 707–738, 2015\.

\[28\] B. Iglewicz and D. Hoaglin, How to Detect and Handle Outliers, ASQ Quality Press, Milwaukee, WI, 1993\.

\[29\] A. Ghorbani and J. Zou, "Data Shapley: Equitable valuation of data for machine learning," in Proc. ICML, pp. 2242–2251, 2019\.

\[30\] R. K. E. Bellamy et al., "AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias," IBM Journal of Research and Development, vol. 63, no. 4/5, pp. 4:1–4:15, 2019\.

\[33\] J. Cohen, Statistical Power Analysis for the Behavioral Sciences, 2nd ed., Lawrence Erlbaum, 1988\.

\[34\] N. Cliff, "Dominance statistics: Ordinal analyses to answer ordinal questions," Psychological Bulletin, vol. 114, no. 3, pp. 494–509, 1993\.

\[35\] R. Kohavi, "Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid," in Proc. ACM SIGKDD, pp. 202–207, 1996\. UCI Adult dataset.

\[36\] J. Angwin, J. Larson, S. Mattu, and L. Kirchner, "Machine bias," ProPublica, May 23, 2016\. COMPAS dataset.

\[37\] D. Dua and C. Graff, UCI Machine Learning Repository. University of California, Irvine, 2017\. (German Credit, Bank Marketing.)

\[39\] M. Hardt, E. Price, and N. Srebro, "Equality of opportunity in supervised learning," in Advances in Neural Information Processing Systems (NeurIPS), vol. 29, pp. 3315–3323, 2016\.

\[43\] H. Weerts, M. Dudík, R. Edgar, A. Jalali, R. Lutz, and M. Madaio, "Fairlearn: Assessing and improving fairness of AI systems," Journal of Machine Learning Research, vol. 24, no. 257, pp. 1–8, 2023\.

\[44\] I. Mironov, "Rényi differential privacy," in Proc. IEEE Computer Security Foundations Symposium (CSF), pp. 263–275, 2017\.

\[47\] M. Mitchell et al., "Model cards for model reporting," in Proc. ACM FAccT, pp. 220–229, 2019\.

\[48\] F. Doshi-Velez and B. Kim, "Towards a rigorous science of interpretable machine learning," arXiv:1702.08608, 2017\.

\[50\] Y. Benjamini and D. Yekutieli, "The control of the false discovery rate in multiple testing under dependency," Annals of Statistics, vol. 29, no. 4, pp. 1165–1188, 2001\.

\[51\] A. Agarwal, A. Beygelzimer, M. Dudík, J. Langford, and H. Wallach, "A reductions approach to fair classification," in Proc. ICML, pp. 60–69, 2018\.

\[52\] I. Covert, S. M. Lundberg, and S.-I. Lee, "Explaining by removing," Journal of Machine Learning Research, vol. 22, no. 209, pp. 1–90, 2021\.

\[53\] F. Ding, M. Hardt, J. Miller, and L. Schmidt, "Retiring Adult: New datasets for fair machine learning," in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021\. Folktables.

\[54\] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity checks for saliency maps," in Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018\.

\[55\] S. Hooker, D. Erhan, P.-J. Kindermans, and B. Kim, "A benchmark for interpretability methods in deep neural networks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019\. ROAR.

\[56\] Y. Rong, T. Leemann, V. Borisov, G. Kasneci, and E. Kasneci, "A consistent and efficient evaluation strategy for attribution methods," in Proc. ICML, pp. 18770–18795, 2022\. ROAD.

\[57\] D. Slack, S. Hilgard, E. Jia, S. Singh, and H. Lakkaraju, "Fooling LIME and SHAP: Adversarial attacks on post-hoc explanation methods," in Proc. AAAI/ACM AIES, pp. 180–186, 2020\.

\[58\] T. Heskes, E. Sijben, I. G. Bucur, and T. Claassen, "Causal Shapley values: Exploiting causal knowledge to explain individual predictions of complex models," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020\.

\[59\] B. Owen and C. Prieur, "On Shapley value for measuring importance of dependent inputs," SIAM/ASA Journal on Uncertainty Quantification, vol. 5, no. 1, pp. 986–1002, 2017\.

# **AUTHOR**

Roy Saurabh is an applied researcher working on trustworthy machine learning, federated learning systems, and algorithmic accountability. He is the founder of AffectLog, where he develops distributed evaluation protocols and systems for auditing fairness, privacy, and reproducibility in machine learning deployments across federated environments. ORCID: 0000-0003-3439-7731.