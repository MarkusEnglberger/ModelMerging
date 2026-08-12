#!/bin/bash
#SBATCH --job-name=apr-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clip8_compare-%j.out
#SBATCH --error=logs/clip8_compare-%j.out

# Full 8-task CLIP fair sweep at the corrected main setting (steps=5, gamma=1,
# clip-after-lr), mirroring slurm/glue8_compare.sbatch. lr grids narrowed from the
# 3-task CLIP findings (APR~32, nogate~8, GD~1e-4); optima drop with task count, so
# each grid also brackets lower. Same replay buffers / merge / #grad-evals per family.
# Requires: python scripts/prefetch_clip.py --all
source slurm/common.sh
python scripts/compare_baselines.py --config configs/clip8.yaml \
  --steps 5 --clip_mode vdist_pre --gammas 1.0 \
  --apr_lrs 8 16 32 64 \
  --nogate_lrs 2 4 8 16 \
  --gd_lrs 5e-5 1e-4 5e-4 \
  --out results/compare/clip8_compare.json "$@"
