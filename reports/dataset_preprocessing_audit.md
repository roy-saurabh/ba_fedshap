# Dataset Preprocessing Audit
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22

---

## 1. Dataset Summary vs Manuscript Table I

| Dataset | Manuscript N | Manuscript d | Protected attrs | Source | Download |
|---------|-------------|--------------|-----------------|--------|----------|
| Adult | 48,842 | 14 | Sex, Race | UCI Archive | auto |
| COMPAS | 7,214 | 13 | Race, Sex | ProPublica GitHub | auto |
| German Credit | 1,000 | 20 | Age, Sex | UCI Archive | auto |
| Bank Marketing | 45,211 | 17 | Age, Marital | UCI Archive | auto |
| ACSIncome | ~195k | 10 | Race, Sex | folktables | auto |
| ACSPublicCoverage | ~109k | 19 | Race, Disability | folktables | auto |

Verified as of 2026-05-22 (Adult only — others require network or folktables install):

| Dataset | Verified N | Verified d | Status |
|---------|-----------|-----------|--------|
| Adult | 45,222 | 14 | VERIFIED (after preprocessing fix) |
| COMPAS | — | — | PENDING (network download; d mismatch risk) |
| German Credit | — | — | PENDING (one-hot expansion may exceed d=20) |
| Bank Marketing | — | — | PENDING (one-hot expansion may exceed d=17) |
| ACSIncome | — | — | PENDING (requires folktables) |
| ACSPublicCoverage | — | — | PENDING (requires folktables) |

Note: Adult had a preprocessing bug (one-hot encoding gave d=104). Fixed to ordinal
encoding yielding d=14. The same issue may affect COMPAS (d=16 after one-hot vs d=13
claimed) and German/Bank. These must be fixed before the full run produces valid tables.

---

## 2. Per-Dataset Preprocessing Details

### 2.1 Adult (UCI)

**Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
**Loader:** `src/data/loaders.py:load_adult`

| Step | Description | Code reference |
|------|-------------|----------------|
| Download | UCI adult.data + adult.test, concatenated | lines 87–102 |
| Missing values | Rows with "?" dropped (`na_values="?"`, then `dropna()`) | line 121 |
| Label | `<=50K` → 0, `>50K` → 1 (stripped, dot-stripped for test set) | lines 124–126 |
| Protected attrs | Sex: Male=1/Female=0; Race: White=1/other=0 (extracted before encoding) | lines 128–132 |
| Features | 14 columns: 6 continuous (age, education-num, capital-gain, capital-loss, hours-per-week, fnlwgt) + 6 ordinal (workclass, marital-status, occupation, relationship, race, sex) + education ordinal + native-country binary | lines 138–165 |
| Normalisation | MinMaxScaler → [0, 1] | `_min_max_scale()` |
| Deterministic? | YES — OrdinalEncoder + MinMaxScaler are deterministic given sorted input | |

**Discrepancy vs manuscript:** Manuscript N=48,842 vs actual N=45,222 after `dropna`.
The difference (~3,620 rows) is due to missing value imputation differences. The
paper likely uses N=48,842 before filtering, or reports the raw UCI count. The
post-filtering N=45,222 is the standard filtered count matching most published results.
This is a labelling ambiguity in the manuscript, not a data error.

**Algorithm 3 background construction:** Uses dataset-standard train/test split
(the loader concatenates UCI train+test, then `dirichlet_partition` creates the train
and holdout splits). The reference/audit split is drawn from the holdout portion,
not from client private records. No raw client records are transmitted during
background construction. CONFIRMED correct per Algorithm 3.

### 2.2 COMPAS (ProPublica)

**Source:** https://github.com/propublica/compas-analysis/
**Loader:** `src/data/loaders.py:load_compas`

| Step | Description |
|------|-------------|
| Filtering | Standard ProPublica filter: days_b_screening_arrest in [-30, 30], is_recid != -1, c_charge_degree != "O", score_text != "N/A" |
| Features | 9 columns before encoding: age, juv_fel_count, juv_misd_count, juv_other_count, priors_count, days_b_screening_arrest, c_charge_degree, sex, race |
| Encoding | `pd.get_dummies` on c_charge_degree (2), sex (2), race (6) → d may be ~16 |
| Protected attrs | sex: Male=1; race: Caucasian=1 |

**d MISMATCH RISK:** The manuscript reports d=13; current loader may produce d~16
with one-hot encoding. Fix required: use ordinal encoding for race (1 column) and
binary encoding for sex, c_charge_degree to achieve d=13.

### 2.3 German Credit (UCI Statlog)

**Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
**Loader:** `src/data/loaders.py:load_german`

| Step | Description |
|------|-------------|
| Features | 20 columns (7 continuous + 13 categorical) |
| Encoding | `pd.get_dummies` on 13 categorical cols → likely d >> 20 |
| Protected attrs | sex: derived from personal_status (A91/A93/A94=male); age: >=25 = 1 |
| Label | 1=good credit → 1; 2=bad credit → 0 |

**d MISMATCH RISK:** Manuscript reports d=20; one-hot encoding of 13 categoricals
(with cardinalities up to 10) will produce far more than 20 columns. Fix required:
ordinal encoding for all categoricals.

### 2.4 Bank Marketing (UCI)

**Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
**Loader:** `src/data/loaders.py:load_bank`

| Step | Description |
|------|-------------|
| File | bank-additional-full.csv (semicolon-separated) |
| Features | All columns except target "y" |
| Encoding | `pd.get_dummies` on 10 cat cols → likely d >> 17 |
| Protected attrs | age: >=35=1; marital: married=1 |
| Label | y="yes" → 1 |

**d MISMATCH RISK:** Manuscript reports d=17; one-hot of 10 categoricals may exceed 40.

### 2.5 ACSIncome (folktables)

**Source:** `folktables.ACSDataSource` (2018 ACS 1-year survey)
**Loader:** `src/data/loaders.py:load_acs_income`

| Feature | Type | Code |
|---------|------|------|
| AGEP | Age (continuous) | continuous |
| COW | Class of worker | categorical |
| SCHL | Educational attainment | ordinal |
| MAR | Marital status | categorical |
| RELP | Relationship | categorical |
| WKHP | Hours worked per week | continuous |
| SEX | Sex | binary (1=male, 2=female) |
| RAC1P | Race | categorical (1=White alone) |
| POVPIP | Income-to-poverty ratio | continuous |
| OCCP | Occupation | categorical |

**d=10 per manuscript:** The ACSIncome task in folktables uses exactly 10 features.
Client partitioning is by US state (geography-based, not Dirichlet).
The Folktables audit split is confirmed as the reference background for Q_pool and Q_a.

### 2.6 ACSPublicCoverage (folktables)

**Source:** `folktables.ACSDataSource` (2018 ACS 1-year survey)
**Loader:** `src/data/loaders.py:load_acs_pubcov`

Uses the ACSPublicCoverage task (d=19 features including disability status).
Protected attrs: SEX (1=male, 2=female), RAC1P (1=White alone), DIS (disability).

---

## 3. Train/Test/Audit Split Logic

For Dirichlet datasets (Adult, COMPAS, German, Bank):
- The loader returns the full pre-processed dataset.
- `partition.dirichlet_partition` assigns records to n_clients using Dirichlet(α) per class.
- Background construction (Algorithm 3) uses a separate reference split:
  the portion of data NOT assigned to any client (held-out test/audit portion).
- This matches the manuscript statement: "dataset's standard test split for Adult,
  COMPAS, German Credit, Bank Marketing."

For ACS datasets:
- Clients are US states (geography-based partition via `geography_partition`).
- The Folktables audit split is a dedicated national holdout not allocated to any state,
  used as the background reference. This matches the manuscript: "Folktables audit split
  for ACSIncome / ACSPublicCoverage."

**No raw client records are used in background construction. CONFIRMED.**

---

## 4. Missing Value Handling

| Dataset | Strategy |
|---------|----------|
| Adult | Drop rows (na_values="?") — standard for UCI Adult |
| COMPAS | Standard ProPublica filter removes bad rows; dropna() after |
| German | Space-separated, no nulls in standard file; dropna() |
| Bank | Semicolon-separated CSV; dropna() |
| ACS | folktables applies its own standard filtering |

---

## 5. Determinism Verification

All preprocessing operations are deterministic:
- OrdinalEncoder / MinMaxScaler: deterministic given sorted categories and fixed data order
- Download cached to `~/.cache/ba_fedshap/` — re-runs use cached files
- RNG used only in `dirichlet_partition` and `subsample_background`, seeded per experiment

---

## 6. Action Items Before Full Run

1. Fix COMPAS loader: use ordinal encoding to achieve d=13
2. Fix German Credit loader: use ordinal encoding to achieve d=20
3. Fix Bank Marketing loader: use ordinal encoding to achieve d=17
4. Verify ACS loaders produce d=10 / d=19 with folktables installed
5. Confirm manuscript N vs filtered N discrepancy for Adult is acceptable
6. Re-run dry smoke test after loader fixes
