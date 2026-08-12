#!/bin/bash
#SBATCH --job-name=apr-cmpfair
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/comparefair-%j.out
#SBATCH --error=logs/comparefair-%j.out

source slurm/common.sh
# Same hyperparameter search budget for every family:
#   APR / no-gate: sweep lr in {1,2,4,8,16} x gamma in {0.5,1.0}
#   ordinary GD  : sweep lr around its known-good region
python scripts/compare_baselines.py --config configs/poc3.yaml \
  --steps 5 \
  --anchored_lrs 1 2 4 8 16 \
  --gammas 0.5 1.0 \
  --gd_lrs 1e-4 5e-4 1e-3 2e-3 \
  --out results/compare/poc3_compare_fair.json "$@"
