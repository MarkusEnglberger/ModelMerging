#!/bin/bash
#SBATCH --job-name=gdmx20-apr30b
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gdmx20_apr30b-%j.out
#SBATCH --error=logs/gdmx20_apr30b-%j.out

# COMPLETES THE BUDGET-SYMMETRIC MATRIX: APR at S=30 (constant + cosine/random)
# from the three inits that never got long-APR arms -- TA, DARE-TIES, and
# Breadcrumbs. GD S=30 already exists for all six inits, so until this runs the
# any-budget comparison at these three pits S30-GD against S5-APR (biased
# against APR). Everywhere long-APR was run it gained (+.004..+.016 mean, up to
# +.12 worst), so expect similar lifts here. Inits pinned to their bracketed
# 20-task optima; replay_obj recorded per cell (post-hoc replay selection).
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0.08 \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.4 \
  --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
  --skip_families ties dare ls ada \
  --refine_from ta dareties bc \
  --apr_lrs 1 2 4 8 \
  --apr_schedules constant cosine --apr_order random \
  --gd_lr_min_frac 0.05 \
  --steps 30 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_apr30b.json "$@"
