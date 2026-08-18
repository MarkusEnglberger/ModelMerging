#!/bin/bash
#SBATCH --job-name=apr-clip20
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/clip20_compare-%j.out
#SBATCH --error=logs/clip20_compare-%j.out

# Full 20-task CLIP fair sweep (TALL-masks suite), same corrected main setting as
# clip8 (steps=5, clip-after-lr). Optimal lr drops with task count
# (3->8 tasks moved APR 32->8), so the APR grid brackets below the 8-task lr=8.
# lam is still the untuned 0.3 from clip8.yaml -- retune (expect smaller, ~0.1)
# before trusting absolute numbers; the merge degrades harder at 20 tasks.
# Requires: python scripts/prefetch_clip.py --suite 20 --verify
# The three webdataset tasks (pcam/fer2013/rendered_sst2) use explicit
# clip-benchmark classnames -- confirm expert acc >> zero-shot in the prefetch
# --verify / expert-eval log before trusting their label order.
source slurm/common.sh
python scripts/compare_baselines.py --config configs/clip20.yaml \
  --steps 5 --clip_mode vdist \
  --apr_lrs 2 4 8 16 \
  --nogate_lrs 1 2 4 8 \
  --gd_lrs 5e-5 1e-4 5e-4 \
  --out results/compare/clip20_compare.json "$@"
