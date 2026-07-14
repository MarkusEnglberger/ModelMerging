#!/bin/bash
#SBATCH --job-name=apr-mrg-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_glue8-%j.out
#SBATCH --error=logs/merge_glue8-%j.out

# Merge baselines on GLUE-8: task arithmetic (lambda swept), TIES, DARE (control),
# DARE-TIES, Model Breadcrumbs, AdaMerging (task+layer wise) -- plus APR refinement
# on top of each init. Same experts / replay buffers / heads as slurm/glue8_compare.sbatch.
#
# APR lr grid {2,4,8,16}: LOWERED and shared across inits. APR's step and trust region
# both scale with |v_i| = |theta_i - theta_init|, which is init-dependent, so the grid
# centred on the TA optimum (16/32) diverged for the TIES/DARE inits in job 6124027.
# lr32 is also knife-edge on GLUE-8 (see the reproducibility note) so it is dropped.
# --mem 48G (up from 32G): TIES/DARE-TIES/Breadcrumbs build extra CPU task-vector copies.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/glue8.yaml \
  --ta_lams 0.2 0.3 0.4 0.5 \
  --ties_densities 0.1 0.2 --ties_lams 0.8 1.0 \
  --dare_densities 0.3 --dare_seeds 0 1 \
  --dt_drops 0.5 --dt_trims 0.1 0.2 --dt_lams 0.8 1.0 \
  --bc_densities 0.1 0.2 --bc_outliers 0.01 0.05 --bc_lams 0.4 \
  --ada_variants task layer --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 --ada_workers 0 \
  --refine_from ta ties dareties bc ada --apr_lrs 2 4 8 16 --steps 5 \
  --out results/compare/merge_baselines2_glue8.json "$@"
