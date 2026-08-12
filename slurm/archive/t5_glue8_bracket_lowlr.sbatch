#!/bin/bash
#SBATCH --job-name=t5-lowlr
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/t5_lowlr-%j.out
#SBATCH --error=logs/t5_lowlr-%j.out
#
# Downward extension of the S=20 grids, which the first pass left edge-clipped
# for the anchored arms.
#
# At S=20 (t5_glue8_bracket_S20.json) APR's arc is monotonically DECREASING over
# the whole grid -- 32:+0.651  64:+0.649  128:+0.644  256:+0.633 -- so its
# optimum lies below lr=32, the grid minimum, and +0.6506 is only a lower bound.
# That matters because ordinary GD peaked interior at +0.7407 in the same run,
# and "GD beats the anchored arms at S=20" cannot be claimed while the arm it
# beats is unbracketed.
#
# The direction is predictable from the S=5 result: the optima track a roughly
# constant lr x S product. APR peaked at lr=128 with S=5 (product 640) and wants
# <=32 at S=20 (32 x 20 = 640); GD peaked at 0.01 with S=5 (0.05) and 0.003 at
# S=20 (0.06). So the S=20 grids should sit ~4x lower than the S=5 grids, and
# this job moves them there.
#
# GD is NOT re-run: it already peaked interior at 0.003 within {0.001 .. 0.03}.
# Extending only the arms that are edge-clipped keeps the comparison honest --
# the arm under test keeps the coverage it already had.
#
# COST: eval0 (3) + TA@l0.3 (1) + 8 cells = 12 passes at ~5.4 min plus S=20
# refinement ~= 1.5-2 h.

set -euo pipefail
source slurm/common.sh

: "${NPROBE:=}"
if [ -n "${NPROBE}" ]; then
  OUT="results/compare/t5_glue8_bracket_S20_lowlr_n${NPROBE}.json"
  NPROBE_ARG=(--n_probe "${NPROBE}")
else
  OUT="results/compare/t5_glue8_bracket_S20_lowlr.json"
  NPROBE_ARG=()
fi
echo "[job] S=20 low-lr extension, n_probe=${NPROBE:-config} -> ${OUT}"

python scripts/merge_baselines.py \
  --config configs/t5_glue8.yaml \
  --ta_lams 0.3 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --steps 20 \
  "${NPROBE_ARG[@]}" \
  --apr_lrs 4 8 16 32 \
  --nogate_lrs 2 4 8 16 \
  --out "${OUT}"
