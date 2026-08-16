#!/bin/bash
#SBATCH --job-name=labeled-n16
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/labeled-n16-%j.out
#SBATCH --error=logs/labeled-n16-%j.out

# Standalone Fisher merging and learned-mask Localize-and-Stitch for the strict
# 16+16 tier. No APR, ungated update, or ordinary GD is run on either baseline.

set -euo pipefail
source slurm/common.sh

: "${BENCH:?set BENCH (glue8|clip8|clip20|t5_glue8)}"

case "$BENCH" in
  glue8)    CFG=configs/glue8.yaml ;;
  clip8)    CFG=configs/clip8.yaml ;;
  clip20)   CFG=configs/clip20_l008.yaml ;;
  t5_glue8) CFG=configs/t5_glue8.yaml ;;
  *) echo "unknown BENCH=$BENCH" >&2; exit 2 ;;
esac

OUT="results/compare/grid_nn_${BENCH}_labeled_n16.json"
python scripts/labeled_baselines_n16.py \
  --config "$CFG" --n_probe 16 --n_select 16 --out "$OUT"
