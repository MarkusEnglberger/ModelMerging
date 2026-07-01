#!/bin/bash
#SBATCH --job-name=apr-clipcmp
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/clip_compare-%j.out
#SBATCH --error=logs/clip_compare-%j.out

# Fair CLIP/ViT baseline sweep at the corrected main setting (steps=5, gamma=1,
# clip-after-lr), mirroring slurm/glue8_compare.sbatch. Same replay buffers, same
# merge point, same #grad-evals for every family; only the update rule differs.
# CLIP per-coord grad scale ~= GLUE's, so the GLUE-tuned lr grids transfer; lr=8
# added on the APR side as a low-end safety margin (||v||~1.9 < GLUE's ~5).
# Requires: python scripts/prefetch_clip.py --tasks eurosat gtsrb mnist
source slurm/common.sh
python scripts/compare_baselines.py --config configs/clip_poc.yaml \
  --steps 5 --clip_mode vdist_pre --gammas 1.0 \
  --apr_lrs 8 16 32 64 128 256 \
  --nogate_lrs 2 4 8 16 32 \
  --gd_lrs 1e-4 5e-4 1e-3 2e-3 5e-3 \
  --out results/compare/clip_poc_compare.json "$@"
