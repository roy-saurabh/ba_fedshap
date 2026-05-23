#!/usr/bin/env bash
# create_release.sh — Run after experiments complete and gh auth login is done.
# Usage: bash scripts/create_release.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Step 1: Verify experiment outputs exist ==="
SEED789="results/raw/compas/seed_789/alpha_0.50/results.json"
SEED1024="results/raw/compas/seed_1024/alpha_0.50/results.json"
for f in "$SEED789" "$SEED1024"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $f — wait for experiment to complete first."
        exit 1
    fi
    echo "  FOUND: $f"
done

echo ""
echo "=== Step 2: SHA-256 manifest ==="
.venv/bin/python3 - <<'EOF'
import hashlib, json
from pathlib import Path

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

manifest = {str(f): {"sha256": sha256(f), "size_bytes": f.stat().st_size}
            for d in ["results/processed", "results/tables", "results/raw"]
            for f in sorted(Path(d).rglob("*.json"))
            if f.stat().st_size > 0}

with open("results/manifest_sha256.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest: {len(manifest)} files -> results/manifest_sha256.json")
EOF

echo ""
echo "=== Step 3: Create compas_raw_results.zip ==="
zip -r compas_raw_results.zip results/raw/compas/
echo "  Created: compas_raw_results.zip ($(du -sh compas_raw_results.zip | cut -f1))"

echo ""
echo "=== Step 4: Commit new files ==="
git add results/raw/compas/ results/manifest_sha256.json \
        CITATION.cff .zenodo.json pyproject.toml .gitignore
git commit -m "feat: add COMPAS raw results for seeds 789 and 1024, SHA-256 manifest, CITATION.cff"

echo ""
echo "=== Step 5: Tag release ==="
git tag -a v1.0.1-softx -m "SoftwareX submission v1.0.1 — BA-FedSHAP reference implementation"

echo ""
echo "=== Step 6: Push to GitHub ==="
echo "  (requires: gh repo create roy-saurabh/ba_fedshap --public --source=. --remote=origin --push)"
read -p "  GitHub repo ready? (y/N): " yn
if [[ "$yn" != "y" && "$yn" != "Y" ]]; then
    echo "  Skipping push. Run manually when ready."
    echo "    gh repo create roy-saurabh/ba_fedshap --public --source=. --remote=origin --push"
    echo "    git push origin v1.0.1-softx"
    exit 0
fi
git push -u origin main
git push origin v1.0.1-softx

echo ""
echo "=== Step 7: Create GitHub release ==="
gh release create v1.0.1-softx \
    --title "v1.0.1-softx — BA-FedSHAP SoftwareX Reference Release" \
    --notes "$(cat <<'NOTES'
## BA-FedSHAP v1.0.1-softx

Reference implementation for the SoftwareX article:
> Roy Saurabh. "BA-FedSHAP: Stabilizing Removal-Based Shapley Attributions for Federated Bias Auditing under Non-IID Data." *SoftwareX*, 2026.

### Contents
- Full BA-FedSHAP protocol (Algorithms 1–3) with RDP accounting and six baselines
- Low-compute COMPAS study reproducer (`configs/compas_lowcompute_softx.yaml`)
- `compas_raw_results.zip` — 15-cell raw JSON corpus (5 seeds × 3 α levels), SHA-256: 378a6c5b91c2a760aacb6a88b4ebab48b69f3491468375067545f264d88186bb
- SHA-256 manifest at `results/manifest_sha256.json`

### Reproduce SoftwareX COMPAS study (~20 min, CPU)
\`\`\`bash
python scripts/run_experiment.py --config configs/compas_lowcompute_softx.yaml --seeds 42 123 456 789 1024
python scripts/make_tables.py --table iv
\`\`\`

### Zenodo DOI
Permanently archived at https://doi.org/10.5281/zenodo.20356218
NOTES
)" \
    compas_raw_results.zip

echo ""
echo "=== Done ==="
echo "Next step: Trigger Zenodo new version via https://zenodo.org/account/settings/github/"
echo "  1. Connect your GitHub account to Zenodo"
echo "  2. Enable the roy-saurabh/ba_fedshap repository"
echo "  3. Zenodo will auto-create a new version for the v1.0.1-softx release"
