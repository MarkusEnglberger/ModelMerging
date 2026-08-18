#!/bin/bash
#SBATCH --job-name=apr-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/glue8-%j.out
#SBATCH --error=logs/glue8-%j.out
source slurm/common.sh
# Full GLUE-8 fair sweep at the corrected main setting (steps=5, clip-after-lr).
# Reduced lr grids informed by the 3-task findings (APR~32, nogate~8, GD~1e-3).
python scripts/compare_baselines.py --config configs/glue8.yaml \
  --steps 5 --clip_mode vdist \
  --apr_lrs 16 32 64 128 256 \
  --nogate_lrs 2 4 8 16 32 \
  --gd_lrs 5e-4 1e-3 2e-3 \
  --out results/compare/glue8_compare.json
