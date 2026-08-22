#!/bin/bash
#SBATCH --job-name=mb3b-brk-rnd
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/mb3b_apr_bracket_random-%A_%a.out
#SBATCH --error=logs/mb3b_apr_bracket_random-%A_%a.out
#
# MergeBench-3B: re-run the APR cells whose split-selected optimum sat on the
# edge of the S={1,2,3,5} x lr<=2048 grid, with the grid extended so the
# optimum can be bracketed (interior in both lr and S):
#
#   0  pretrained, n=16+16  (was lr=2048, S=5: top edge on BOTH axes)
#   1  pretrained, n= 8+ 8  (was lr=1024, S=5: S at the top edge)
#   2  TA (lam=0.5), n=8+8  (was lr=128,  S=5: S at the top edge)
#
# Cells that were already interior (BC/DARE-TIES/TIES/Ada compositions, GD)
# are not re-run. Low learning rates and S=1, which were dominated everywhere
# in the previous grids, are dropped so the search stays the same size as
# GD's (30 vs 32 cells) while reaching further up on both axes.
#
# Task order is re-randomized at every sweep (project convention from here
# on); constant learning rate; no annealing, no GD, no ungated arm. Every
# hyperparameter is chosen on the disjoint selection buffer.

set -euo pipefail
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEPS="2 3 5 8 12"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG=pretrained_n16; N=16
    LRS="256 512 1024 2048 4096 8192"
    FAMILY_ARGS=(--skip_families ties dare dareties della bc ls ada
                 --ta_lams 0 --refine_from ta)
    ;;
  1)
    TAG=pretrained_n8; N=8
    LRS="256 512 1024 2048 4096 8192"
    FAMILY_ARGS=(--skip_families ties dare dareties della bc ls ada
                 --ta_lams 0 --refine_from ta)
    ;;
  2)
    TAG=ta_n8; N=8
    LRS="32 64 128 256 512"
    FAMILY_ARGS=(--skip_families ties dare dareties della bc ls ada
                 --ta_lams 0.5 --refine_from ta)
    ;;
  *)
    echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT="results/compare/grid_nn_mb3b_${TAG}_apr_s2to12_random.json"
echo "[job] ${TAG} n=${N}+${N} S={${STEPS}} lr={${LRS}} random-order APR -> ${OUT}"

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --n_probe "${N}" --n_select "${N}" \
  --steps ${STEPS} \
  --select_by mean_acc \
  --apr_schedules constant \
  --apr_constant_order random \
  --apr_lrs ${LRS} \
  --nogate_lrs \
  --gd_lrs \
  --control_gd_lrs \
  --out "${OUT}" \
  "${FAMILY_ARGS[@]}"

echo "[done] -> ${OUT}"
