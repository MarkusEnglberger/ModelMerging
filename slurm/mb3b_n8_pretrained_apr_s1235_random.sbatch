#!/bin/bash
#SBATCH --job-name=mb3b-pre-rnd
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=logs/mb3b_n8_pretrained_apr_s1235_random-%j.out
#SBATCH --error=logs/mb3b_n8_pretrained_apr_s1235_random-%j.out
#
# MergeBench-3B n=8+8 pretrained follow-up. Lambda zero pins task arithmetic
# exactly to the pretrained checkpoint. APR uses a constant learning rate and
# independently reshuffles the task order at every sweep.
#
# Intentionally absent: annealing, ordinary GD, and ungated APR.

set -euo pipefail
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="results/compare/grid_nn_mb3b_pretrained_n8_apr_s1235_random.json"
echo "[job] pretrained n=8+8 S={1,2,3,5} random-order APR -> ${OUT}"

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --n_probe 8 --n_select 8 \
  --steps 1 2 3 5 \
  --select_by mean_acc \
  --skip_families ties dare dareties della bc ls ada \
  --ta_lams 0 \
  --refine_from ta \
  --apr_schedules constant \
  --apr_constant_order random \
  --apr_lrs 16 32 64 128 256 512 1024 2048 \
  --nogate_lrs \
  --gd_lrs \
  --control_gd_lrs \
  --out "${OUT}"

echo "[done] -> ${OUT}"
