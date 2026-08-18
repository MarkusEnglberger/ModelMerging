#!/bin/bash
#SBATCH --job-name=apr-budget
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/budget-%j.out
#SBATCH --error=logs/budget-%j.out
source slurm/common.sh
python scripts/budget_sweep.py --config configs/poc3.yaml \
  --steps 2 \
  --n_probes 8 16 32 64 --seeds 0 1 2 \
  --out results/compare/budget_sweep.json "$@"
