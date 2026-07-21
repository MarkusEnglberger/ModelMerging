#!/bin/bash
#SBATCH --job-name=gdmx20-la
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/gdmx20_long_a-%j.out
#SBATCH --error=logs/gdmx20_long_a-%j.out

# LONG-HORIZON + ANNEALED ordinary GD from the TA and TIES inits.
# Companion to gd_matrix_clip20_s5 (matched-budget S=5). Tests the standard
# objection "GD just needs more sweeps": S=30 (6x the APR budget) crossed with
# constant vs cosine annealing (to 5% of peak lr, random task order), which
# SEPARATES the two explanations -- more sweeps alone vs annealing. Includes
# aggressive lrs (5e-4, 1e-3) that DIVERGED at constant lr from TA, precisely to
# test whether annealing rescues them. On clip8 from TA, GD at S=35 reached .526
# vs APR .615 at S=5 and never caught up; this checks that at 20 tasks.
# APR/nogate reference cells live in the S=5 matrix report (same pinned inits).
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0.08 --ties_densities 0.1 --ties_lams 0.6 \
  --skip_families dare dareties bc ada \
  --refine_from ta ties \
  --apr_lrs \
  --gd_lrs 5e-5 1e-4 5e-4 1e-3 \
  --gd_steps 30 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 5 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_long_a.json "$@"
