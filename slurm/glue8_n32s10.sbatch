#!/bin/bash
#SBATCH --job-name=apr-g8n32
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/glue8n32-%j.out
#SBATCH --error=logs/glue8n32-%j.out
source slurm/common.sh
# GLUE-8 variant: n_probe=32 (fewer samples), steps=10 (more sweeps). Same corrected
# setting (gamma=1, clip-after-lr) and per-family lr grids.
python scripts/compare_baselines.py --config configs/glue8.yaml \
  --steps 10 --n_probe 32 --clip_mode vdist_pre --gammas 1.0 \
  --apr_lrs 16 32 64 128 256 \
  --nogate_lrs 2 4 8 16 32 \
  --gd_lrs 5e-4 1e-3 2e-3 \
  --out results/compare/glue8_n32s10_compare.json
