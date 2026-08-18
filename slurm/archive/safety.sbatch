#!/bin/bash
#SBATCH --job-name=apr-safety
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/safety-%j.out
#SBATCH --error=logs/safety-%j.out
source slurm/common.sh
python scripts/safety_sweep.py --config configs/poc3.yaml --steps 5 \
  --out results/compare/safety_sweep.json "$@"
