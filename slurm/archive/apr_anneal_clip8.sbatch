#!/bin/bash
#SBATCH --job-name=apr-anneal-c8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/apr_anneal_clip8-%j.out
#SBATCH --error=logs/apr_anneal_clip8-%j.out
#
# Does offering APR the ANNEALED schedule change the n+n answer on CLIP-8?
#
# Every n=16+16 grid cell in the paper ran APR at a constant rate and FIXED task
# order: merge_baselines.py only consults --apr_order when the schedule is not
# constant, and the configs set refine.order=fixed, so the "apr_order: random"
# recorded in those reports never took effect. The long-horizon control at n=64
# (Table tab:longapr) found the annealed package -- cosine decay to 5% of the
# peak, task order re-randomized every sweep -- worth a lot on GLUE-8, so the
# main grid may be under-searching APR. This job tests that on CLIP-8.
#
# Scope is deliberately narrow: APR only (no ungated, no GD arm), and only the
# two initializations asked for. The controls do not need re-running -- the
# question is whether APR has been given its best configuration, not whether the
# baselines have.
#
# Both schedules run in ONE report so split selection can choose across their
# union on the selection buffer. Cell names carry a ",cosine" suffix, and each
# schedule's own winner is evaluated, so the union winner is always one of the
# evaluated cells.
#
# Learning rates go one octave above the constant grid. Cosine decaying to 5%
# spends roughly 0.53x the peak rate on average, so the annealed optimum should
# sit near 2x the constant optimum (lr=8 from both inits); 64 keeps that
# bracketed instead of clipping at the edge.
#
# The two merge cells are PINNED to the values the matched-budget run already
# split-selected (TA lam=0.3, TIES d=0.1 lam=1.0). Family selection is
# deterministic given the same buffers and seeds, so re-deriving them would
# return the same cells at the cost of eight full-eval merge cells.

set -euo pipefail
source slurm/common.sh

OUT=results/compare/grid_nn_clip8_merges_n16_anneal.json

python scripts/merge_baselines.py \
  --config configs/clip8.yaml \
  --n_probe 16 --n_select 16 \
  --steps 5 20 50 \
  --select_by mean_normret \
  --ta_lams 0.3 \
  --ties_densities 0.1 --ties_lams 1.0 \
  --skip_families dare dareties della bc ls ada \
  --refine_from ta ties \
  --apr_lrs 2 4 8 16 32 64 \
  --apr_schedules constant cosine \
  --apr_order random \
  --gd_lr_min_frac 0.05 \
  --out "$OUT"

echo "[done] -> $OUT"
