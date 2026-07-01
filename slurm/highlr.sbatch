#!/bin/bash
#SBATCH --job-name=apr-highlr
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/highlr-%j.out
#SBATCH --error=logs/highlr-%j.out
source slurm/common.sh
python scripts/highlr_diag.py --config configs/poc3.yaml --steps 5 --gamma 0.5 \
  --out results/compare/highlr_diag.json "$@"
