#!/bin/bash
#SBATCH --job-name=apr-mrg3-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge3_glue8-%j.out
#SBATCH --error=logs/merge3_glue8-%j.out

# Run 3 (GLUE): give Model Breadcrumbs a real tuning budget. In job 6124433 it got a
# single lambda (0.4) and scored 0.097 -- that is an artifact of no tuning, not the method.
# Everything else on GLUE-8 is already bracketed (every APR-on-top init peaks at an
# interior lr4), so only the BC family and APR-from-BC are re-run here.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/glue8.yaml \
  --skip_families ta ties dare dareties ada \
  --bc_densities 0.1 0.2 --bc_outliers 0.01 0.05 --bc_lams 0.1 0.2 0.3 0.4 \
  --refine_from bc --apr_lrs 2 4 8 --steps 5 \
  --out results/compare/merge_baselines3_glue8_bc.json "$@"
