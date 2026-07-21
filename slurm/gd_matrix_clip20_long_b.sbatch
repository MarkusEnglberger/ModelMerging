#!/bin/bash
#SBATCH --job-name=gdmx20-lb
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/gdmx20_long_b-%j.out
#SBATCH --error=logs/gdmx20_long_b-%j.out

# LONG-HORIZON + ANNEALED ordinary GD from the DARE-TIES and Breadcrumbs inits.
# See gd_matrix_clip20_long_a for the rationale: S=30 (6x the APR budget) crossed
# with constant vs cosine annealing separates "more sweeps" from "annealing",
# and the aggressive lrs test whether annealing rescues step sizes that diverge
# at constant lr. Inits pinned to their best-known 20-task configs.
# NOTE: both families' lambdas sat at the LOW GRID EDGE in job 6204114 (job
# 6205649 is bracketing them); if that job moves the optimum, these inits should
# be rebuilt at the new lambda before the numbers go in the paper.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.4 \
  --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
  --skip_families ta ties dare ada \
  --refine_from dareties bc \
  --apr_lrs \
  --gd_lrs 5e-5 1e-4 5e-4 1e-3 \
  --gd_steps 30 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 5 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_long_b.json "$@"
