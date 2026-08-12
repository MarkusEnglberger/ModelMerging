#!/bin/bash
#SBATCH --job-name=apr-clipdiag
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/clipdiag-%j.out
#SBATCH --error=logs/clipdiag-%j.out
source slurm/common.sh
python scripts/clip_diag.py --config configs/poc3.yaml --steps 5 \
  --out results/compare/clip_diag.json "$@"
