#!/bin/bash
#SBATCH --job-name=apr-budgmain
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/budgmain-%j.out
#SBATCH --error=logs/budgmain-%j.out
source slurm/common.sh
# Multi-seed best-of-family confirmation at main setting: n=64, steps=5, gamma=1.
python scripts/budget_sweep.py --config configs/poc3.yaml \
  --steps 5 --gamma 1.0 \
  --n_probes 64 --seeds 0 1 2 3 4 \
  --out results/compare/budget_main_n64s5.json
