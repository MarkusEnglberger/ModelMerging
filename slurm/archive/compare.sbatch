#!/bin/bash
#SBATCH --job-name=apr-compare
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/compare-%j.out
#SBATCH --error=logs/compare-%j.out

source slurm/common.sh
python scripts/compare_baselines.py --config configs/poc3.yaml \
  --steps 5 \
  --gd_lrs 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 \
  --out results/compare/poc3_compare.json "$@"
