#!/bin/bash
#SBATCH --job-name=apgd-nn
#SBATCH --partition=mcs.gpu.q
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/apgd_nn-%j.out
#SBATCH --error=logs/apgd_nn-%j.out
#
# DOGE/APGD (Wei et al., ICML 2025) under the paper's strict n+n protocol.
# The merge itself is data-free.  For each budget, only eta selection reads the
# disjoint selection buffer; the evaluation split is read once for the winner.
# Run n=8 then n=16 in one job so the latter reuses the data-free merge cache.

set -euo pipefail
source slurm/common.sh

: "${BENCH:?set BENCH (glue8|clip8|clip20)}"

case "$BENCH" in
  glue8)  CFG=configs/glue8.yaml;       ETAS="0.075 0.1 0.15 0.2" ;;
  clip8)  CFG=configs/clip8.yaml;       ETAS="0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.18" ;;
  clip20) CFG=configs/clip20_l008.yaml; ETAS="0.0025 0.005 0.0075 0.01 0.015 0.02 0.025 0.03 0.05" ;;
  *) echo "unknown BENCH=$BENCH" >&2; exit 2 ;;
esac

# Optional recovery/edge-audit grid.  This lets a dependent job extend an
# observed boundary hit while retaining the canonical defaults above.
if [[ -n "${ETAS_OVERRIDE:-}" ]]; then
  ETAS="$ETAS_OVERRIDE"
fi
if [[ "${APGD_GRID:-}" == "lower" ]]; then
  ETAS="0.0025 0.005 0.0075 0.01 0.015 0.02 0.025 0.03 0.05"
fi

for N in 8 16; do
  OUT="results/compare/grid_nn_${BENCH}_apgd_n${N}.json"
  echo "[job] DOGE/APGD BENCH=$BENCH n=${N}+${N} eta={$ETAS} -> $OUT"
  python scripts/apgd_baseline.py \
    --config "$CFG" --n_probe "$N" --n_select "$N" \
    --etas $ETAS --out "$OUT"
done
