#!/bin/bash
#SBATCH --job-name=apr-clip8n32
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/clip8_n32-%j.out
#SBATCH --error=logs/clip8_n32-%j.out

# Smaller replay buffer (n=32) run on the 8-task CLIP suite. APR and ordinary GD
# only (no no-gate, no inverted control), each at its two best learning rates from
# the n=64 runs, and with different sweep budgets: APR 10 sweeps, GD 25 sweeps.
# Two invocations because compare_baselines uses a single global --steps.
source slurm/common.sh

echo "===== APR (S=10, n=32, lr in {6,8}) ====="
python scripts/compare_baselines.py --config configs/clip8.yaml \
  --steps 10 --clip_mode vdist_pre --gammas 1.0 --skip_inverted --n_probe 32 \
  --apr_lrs 6 8 \
  --nogate_lrs \
  --gd_lrs \
  --out results/compare/clip8_n32_apr.json

echo "===== Ordinary GD (S=25, n=32, lr in {5e-5,1e-4}) ====="
python scripts/compare_baselines.py --config configs/clip8.yaml \
  --steps 25 --clip_mode vdist_pre --gammas 1.0 --skip_inverted --n_probe 32 \
  --apr_lrs \
  --nogate_lrs \
  --gd_lrs 5e-5 1e-4 \
  --out results/compare/clip8_n32_gd.json
