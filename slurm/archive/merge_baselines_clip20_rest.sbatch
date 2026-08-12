#!/bin/bash
#SBATCH --job-name=apr-mrg20-rest
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/merge_clip20_rest-%j.out
#SBATCH --error=logs/merge_clip20_rest-%j.out

# The merge families SKIPPED in the first CLIP-20 pass (6188401): DARE (control),
# DARE-TIES, Breadcrumbs -- completing the paper's CLIP table at 20 tasks, with
# APR refined from dareties/bc (as in the CLIP-8 run-2). Lambda grids shifted
# DOWN vs clip8 (tuned TA lam fell 0.2->0.08; TIES lam 0.8->0.6), and bc_lams is
# a real sweep this time (the clip8 single-value 0.4 was flagged undertuned).
# Selection/reporting by mean_acc (NormRet degenerate: stl10/food101).
# Requires: prefetch_clip.py --suite 20.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --dare_densities 0.3 --dare_seeds 0 \
  --dt_drops 0.5 --dt_trims 0.1 0.2 --dt_lams 0.4 0.6 0.8 \
  --bc_densities 0.1 0.2 --bc_outliers 0.01 0.05 --bc_lams 0.2 0.4 \
  --skip_families ta ties ada \
  --refine_from dareties bc --apr_lrs 2 4 8 --steps 5 --select_by mean_acc \
  --out results/compare/merge_baselines_clip20_rest.json "$@"
