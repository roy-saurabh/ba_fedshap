# Hash Manifest Report
**BA-FedSHAP IEEE Access Submission**
**Date:** 2026-05-22

The full SHA256 manifest of result files is at:
  results/manifest_sha256.txt
  results/manifest_sha256.json

Currently only smoke-test outputs exist. Final table CSVs will be added after
the full pipeline runs on the pinned hardware.

---

## Source Code Hashes (as of 2026-05-22)

| File | SHA256 | Size |
|------|--------|------|
| src/federated/ba_fedshap.py | 9bc56cd4a54144262751d69cfdad9e5a0d70132a | 16,747 b |
| src/federated/training.py | 568fdfa1d519a41c9313630772dc1e0315a991da | 9,028 b |
| src/explainers/kernelshap_wrapper.py | f7ebabf5e9272ee80318f5b915109a93d4d32ea1 | 4,445 b |
| src/explainers/baselines.py | 8a55c4fef7b943c6fcff86b5082ea404d88f26f3 | 7,506 b |
| src/metrics/adi.py | 218c3d0324e989a0ed1b0afdeae80fc301cd9697 | 7,240 b |
| src/metrics/faithfulness.py | 9df2b17f9a13def3347bef658b10a454ff6686d0 | 5,286 b |
| src/metrics/sanity.py | 7ad65fc38441b7e16b13dc38ac54f4b1d9afc83b | 5,445 b |
| src/metrics/lipschitz.py | 7b84d8c9ee8b512dc3573558acd6ba6aea9afcf2 | 4,086 b |
| src/privacy/rdp_accountant.py | 1e9067571331e3f19f6e523a4b4ba65d64007a69 | 7,956 b |
| src/privacy/dp_noise.py | 7cce711d0e84b9a093f0a7fe3c4f46e74472bfdf | 3,595 b |
| src/data/loaders.py | af7b3ce9878ac8e210af6b1470b1988f38baf265 | 17,268 b |
| src/data/partition.py | 1158807764f9383338c0b52b3e910c98c17b76b5 | 4,408 b |

---

## Config Hashes

| Config | SHA256 | Size |
|--------|--------|------|
| configs/adult.yaml | 378c10a73bd9799ac5d1dbd028d9e2d7224e55ba | 595 b |
| configs/compas.yaml | 76f45c05dd12ab366af526ebb59e5cf486081b89 | 596 b |
| configs/german.yaml | b0dbac4b59f2de1a561bec36f9ea07e81978611a | 595 b |
| configs/bank.yaml | 9376e9a6395438d87ce8a7fb3f8bb7a3cf471632 | 597 b |
| configs/acs_income.yaml | aaf896c95a439780e97c403d572a04f323f67dfe | 707 b |
| configs/acs_pubcov.yaml | 697e750ec0248c4e3e87ccbf53ac48df3c74d426 | 707 b |

---

## Environment Hashes

| File | SHA256 | Size |
|------|--------|------|
| environment.yml | 90f8752b004453d2ab220c1983f0fb89ba293e6c | 457 b |
| requirements.txt | 2c91b1aeca5ead9b7a0663f6081edf0647a085c9 | 266 b |

---

## Smoke Test Output Hashes

| File | SHA256 | Size |
|------|--------|------|
| results/smoke_test/dry_smoke_test_results.json | 32b366567c2b68a1f40ab77f0e304dae4a3d7498 | 1,952 b |

---

## Result Table Hashes (pending full run)

These files will be populated after `bash scripts/run_all_main.sh` completes
on the pinned hardware:

- results/tables/table_iv_primary_reliability.csv — PENDING
- results/tables/table_v_faithfulness_sanity.csv — PENDING
- results/tables/table_vi_ablation.csv — PENDING
- results/tables/table_vii_dp_noise.csv — PENDING
- results/tables/table_viii_adi_outcome_correlation.csv — PENDING
- results/tables/table_ix_lipschitz.csv — PENDING
- results/tables/table_x_runtime.csv — PENDING

---

## How to Regenerate Manifest

```bash
# After full run
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
        manifest[str(f)] = {"sha256": sha256(f), "size": f.stat().st_size}

with open("results/manifest_sha256.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest: {len(manifest)} files")
PYEOF
```
