#!/bin/bash
#SBATCH --job-name=della-apr-clip8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/della_apr_clip8-%j.out
#SBATCH --error=logs/della_apr_clip8-%j.out

# Paper-default DELLA followed by the established CLIP-8 APR learning-rate grid.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip8.yaml \
  --della_densities 0.7 --della_windows 0.14 --della_lams 1.1 --della_seeds 42 \
  --skip_families ta ties dare dareties bc ls ada \
  --refine_from della --apr_lrs 2 3 4 6 8 --steps 5 \
  --out results/compare/della_apr_clip8.json "$@"
