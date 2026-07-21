#!/bin/bash
#SBATCH --job-name=apr-mrg20-restlow
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/merge_clip20_restlow-%j.out
#SBATCH --error=logs/merge_clip20_restlow-%j.out

# Bracket the DARE-TIES and Breadcrumbs lambdas at 20 tasks. In job 6204114 BOTH
# sat on the LOW EDGE of their grids and were monotone increasing as lambda fell:
#   DARETIES  l0.8 .5201 < l0.6 .5810 < l0.4 .6168  (edge of {0.4,0.6,0.8})
#   BC        l0.4 .5876 < l0.2 .6110               (edge of {0.2,0.4})
# so neither is a FAIR baseline yet -- the same complaint already logged against
# the clip8 Breadcrumbs row (single lambda 0.4). DARE-TIES needs the smallest
# lambdas because its 1/density rescale inflates magnitudes. Same low-lambda
# shift seen for TA (0.2->0.08) and TIES (0.8->0.6) at 20 tasks.
# Reporting/selection by mean_acc (NormRet degenerate: stl10/food101).
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --dt_drops 0.5 --dt_trims 0.1 0.2 --dt_lams 0.1 0.2 0.3 \
  --bc_densities 0.1 0.2 --bc_outliers 0.01 0.05 --bc_lams 0.05 0.1 \
  --skip_families ta ties ada dare \
  --refine_from dareties bc --apr_lrs 2 4 8 --steps 5 --select_by mean_acc \
  --out results/compare/merge_baselines_clip20_rest_low.json "$@"
