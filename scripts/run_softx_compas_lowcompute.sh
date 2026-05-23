#!/usr/bin/env bash
set -euo pipefail

# Reproduces all 15 cells (5 seeds × 3 α) from the SoftwareX manuscript Table 1.
# Runtime: ~2 CPU-hours on a standard laptop.
# Output: results/raw/<seed>_<alpha>/full_eval.json

for SEED in 42 123 456 789 1024; do
  for ALPHA in 0.1 0.5 1.0; do
    echo "==> seed=${SEED} alpha=${ALPHA}"
    python scripts/run_full_eval.py \
      --config configs/compas_lowcompute_softx.yaml \
      --seed "$SEED" \
      --alpha "$ALPHA" \
      --results-dir results/raw \
      --n-clients 10 \
      --fl-rounds 10 \
      --coalitions 64 \
      --background-size 200 \
      --skip-sanity \
      --skip-lipschitz
  done
done

echo "Done. Generate Table 1 with:"
echo "  python scripts/make_softx_compas_table.py --input results/raw/compas --output results/tables/table1_compas_aggregate.csv"
echo "Or all manuscript tables with: python scripts/make_paper_tables.py"
