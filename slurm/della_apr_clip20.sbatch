#!/bin/bash
#SBATCH --job-name=della-apr-clip20
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/della_apr_clip20-%j.out
#SBATCH --error=logs/della_apr_clip20-%j.out

# Paper-default DELLA followed by the established CLIP-20 APR grid. Absolute
# mean accuracy is used because several 20-task expert/base gaps are degenerate.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --della_densities 0.7 --della_windows 0.14 --della_lams 1.1 --della_seeds 42 \
  --skip_families ta ties dare dareties bc ls ada \
  --refine_from della --apr_lrs 1 2 4 8 --steps 5 --select_by mean_acc \
  --out results/compare/della_apr_clip20.json "$@"
