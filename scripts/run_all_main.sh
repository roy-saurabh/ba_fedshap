#!/bin/bash
# run_all_main.sh — Full BA-FedSHAP experiment pipeline
# Runs all six dataset configs, then generates tables and figures.
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Activate conda environment if available
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ba-fedshap 2>/dev/null || echo "Warning: could not activate ba-fedshap env"
fi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGDIR="logs/full_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logging to: $LOGDIR"

# ---------------------------------------------------------------------------
# Run experiments for each dataset config
# ---------------------------------------------------------------------------
CONFIGS=(
    configs/adult.yaml
    configs/compas.yaml
    configs/german.yaml
    configs/bank.yaml
    configs/acs_income.yaml
    configs/acs_pubcov.yaml
)

for cfg in "${CONFIGS[@]}"; do
    dataset=$(basename "$cfg" .yaml)
    logfile="$LOGDIR/${dataset}.log"
    echo "========================================"
    echo "Running: $cfg -> $logfile"
    echo "========================================"
    python scripts/run_experiment.py --config "$cfg" 2>&1 | tee "$logfile"
    echo "Done: $dataset"
done

# ---------------------------------------------------------------------------
# Aggregate results and generate outputs
# ---------------------------------------------------------------------------
echo "========================================"
echo "Generating tables..."
echo "========================================"
python scripts/make_tables.py 2>&1 | tee "$LOGDIR/make_tables.log"

echo "========================================"
echo "Generating figures..."
echo "========================================"
python scripts/make_figures.py 2>&1 | tee "$LOGDIR/make_figures.log"

echo "========================================"
echo "Done. Results in results/"
echo "Log: $LOGDIR"
echo "========================================"
