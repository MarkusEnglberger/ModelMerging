#!/bin/bash
#SBATCH --job-name=apr-cmpg1
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cmpg1-%j.out
#SBATCH --error=logs/cmpg1-%j.out
source slurm/common.sh
# Detailed fair sweep at the MAIN setting: n=64, steps=5, clip AFTER lr.
python scripts/compare_baselines.py --config configs/poc3.yaml \
  --steps 5 --clip_mode vdist \
  --apr_lrs 2 4 8 16 32 64 \
  --nogate_lrs 2 4 8 16 32 64 \
  --gd_lrs 1e-4 5e-4 1e-3 2e-3 \
  --out results/compare/poc3_compare_g1.json
