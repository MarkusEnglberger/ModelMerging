#!/bin/bash
#SBATCH --job-name=apr-rpl-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/replay_clip8-%j.out
#SBATCH --error=logs/replay_clip8-%j.out

# Matched-data-budget table on the 8-task CLIP ViT-B/32 suite. Mirrors
# slurm/replay_baselines_glue8.sbatch: every method sees the SAME 64 labeled examples
# per task, and hyperparameters are selected on the replay objective, not the eval split.
#
# APR lr grid {4,6,8,12}: CLIP's APR optimum from the TA merge sits near 8 and the grid
# must bracket it from above (job 6124434 peaked at the lr8 high edge).
# Fisher uses per-example (batch-size-1) backward passes, so it decodes 64 images/task.
# --mem 96G for the same reason as the merge-baselines job (~33G resident footprint).
# Requires: python scripts/prefetch_clip.py --all
source slurm/common.sh
python scripts/replay_baselines.py --config configs/clip8.yaml \
  --n_probe 64 --steps 5 \
  --lam_grid 0.05 0.1 0.15 0.2 0.3 0.4 \
  --ls_sparsities 0.01 0.05 --ls_gamma 1e-4 --ls_steps 200 --ls_lr 0.1 --ls_bs 16 \
  --head_lr 1e-3 --head_epochs 20 \
  --apr_lrs 4 6 8 12 --nogate_lrs 1 2 4 8 --gd_lrs 5e-5 1e-4 5e-4 1e-3 \
  --out results/compare/replay_baselines_clip8.json "$@"
