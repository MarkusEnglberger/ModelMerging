#!/bin/bash
#SBATCH --job-name=apr-clip8n16
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/clip8_n16-%j.out
#SBATCH --error=logs/clip8_n16-%j.out

# Same as slurm/clip8_n32.sbatch but with n=16 replay examples per task.
# APR (S=10, lr in {6,8}) and ordinary GD (S=25, lr in {5e-5,1e-4}) only; no
# no-gate, no inverted control. Two invocations (different --steps per family).
source slurm/common.sh

echo "===== APR (S=10, n=16, lr in {6,8}) ====="
python scripts/compare_baselines.py --config configs/clip8.yaml \
  --steps 10 --clip_mode vdist --skip_inverted --n_probe 16 \
  --apr_lrs 6 8 \
  --nogate_lrs \
  --gd_lrs \
  --out results/compare/clip8_n16_apr.json

echo "===== Ordinary GD (S=25, n=16, lr in {5e-5,1e-4}) ====="
python scripts/compare_baselines.py --config configs/clip8.yaml \
  --steps 25 --clip_mode vdist --skip_inverted --n_probe 16 \
  --apr_lrs \
  --nogate_lrs \
  --gd_lrs 5e-5 1e-4 \
  --out results/compare/clip8_n16_gd.json
