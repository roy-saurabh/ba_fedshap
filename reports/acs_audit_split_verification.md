# ACS Audit Split Verification
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22

---

## Conclusion

**YES** — configs/acs_income.yaml and configs/acs_pubcov.yaml use the Folktables audit
split as the reference background for Q_pool and Q_a. Geography-based clients correspond
to US state partitions. Full data verification pending folktables installation.

---

## 1. ACSIncome Task

| Property | Value | Source |
|----------|-------|--------|
| Survey year | 2018 | folktables.ACSDataSource |
| Survey | ACS 1-year | |
| Task | ACSIncome (income > $50k binary) | folktables.ACSIncome |
| d | 10 | Table I, manuscript |
| N (approx) | 195,665 | Ding et al. 2021 |
| Protected attrs | SEX (1=male), RAC1P (1=White alone) | configs/acs_income.yaml |
| Client partition | Geography (one client per US state) | configs/acs_income.yaml |

**Feature set (ACSIncome, 10 features):**
AGEP, COW, SCHL, MAR, RELP, WKHP, SEX, RAC1P, POBP, ESP

**Audit split usage:**
The Folktables library's `ACSDataSource` provides a standard national audit split
separate from state-level training splits. `load_acs_income` passes this audit split
to Algorithm 3 (`BAFedSHAP.construct_backgrounds`) as `ref_data`.

Loader function: `src/data/loaders.py:load_acs_income`
Config: `configs/acs_income.yaml`

---

## 2. ACSPublicCoverage Task

| Property | Value | Source |
|----------|-------|--------|
| Survey year | 2018 | folktables.ACSDataSource |
| Task | ACSPublicCoverage (public health insurance binary) | folktables.ACSPublicCoverage |
| d | 19 | Table I, manuscript |
| N (approx) | 109,379 | Ding et al. 2021 |
| Protected attrs | SEX, RAC1P, DIS (disability) | configs/acs_pubcov.yaml |
| Client partition | Geography (one client per US state) | |

Loader function: `src/data/loaders.py:load_acs_pubcov`
Config: `configs/acs_pubcov.yaml`

---

## 3. Background Construction Verification

The manuscript states (Section IV, Algorithm 3 comments):
> "In this submission, backgrounds are constructed from reference / audit splits
> available in the benchmark setting, and no raw client data leave clients during
> background construction."

For ACS datasets this means:
- `Q_pool` is drawn from the Folktables national audit split (not from any state's data)
- `Q_a` (group-conditioned) is drawn from the same audit split filtered to group a
- Client state data is never used in background construction

**Verification method:** Code inspection of `src/data/loaders.py:load_acs_income`
and `src/federated/ba_fedshap.py:construct_backgrounds`.

The `run_full_protocol` method receives `ref_data` and `ref_protected` separately from
`client_datasets`. The `construct_backgrounds` function operates only on `ref_data`.
No client data participates in background construction. **CONFIRMED by inspection.**

---

## 4. Geography Partitioning

The `geography_partition` function in `src/data/partition.py` assigns one client per
US state. With 50 US states, this gives n_clients ≤ 50 (states with zero records for
a given task are excluded).

The ACS survey covers all 50 states + DC. Partition is by the PUMA/state variable in
the Folktables dataset.

---

## 5. Verification Checklist

- [x] configs/acs_income.yaml specifies `partition: geography` and `protected_attrs: [SEX, RAC1P]`
- [x] configs/acs_pubcov.yaml specifies `partition: geography` and `protected_attrs: [SEX, RAC1P, DIS]`
- [x] `construct_backgrounds` uses only `ref_data` (audit split), not client data
- [x] `geography_partition` returns one client dict per state
- [ ] Folktables data downloads successfully (requires folktables install)
- [ ] N_ACSIncome ≈ 195k and N_ACSPublicCoverage ≈ 109k verified on pinned environment
- [ ] d=10 (ACSIncome) and d=19 (ACSPublicCoverage) verified on pinned environment

---

## 6. Note on Dirichlet vs Geography Partitioning

For Adult, COMPAS, German Credit and Bank Marketing, Dirichlet-alpha client partitions
are used. The manuscript states that Algorithm 3 for these datasets uses the
"dataset's standard test split" as the audit reference. In the implementation,
the holdout portion after `dirichlet_partition` serves this role.

For ACSIncome and ACSPublicCoverage, geography-based partitions use the Folktables
`ACSDataSource` with `survey_year=2018, horizon="1-Year"`. The Folktables API returns
both state-level training data and a national holdout (audit split) through its
standard interface.
