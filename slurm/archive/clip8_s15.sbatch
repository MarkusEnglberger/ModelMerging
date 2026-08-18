#!/bin/bash
#SBATCH --job-name=apr-clip8s15
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/clip8_s15-%j.out
#SBATCH --error=logs/clip8_s15-%j.out

# Longer-refinement 8-task CLIP run: S=15 sweeps, only the best learning rate per
# family from the S=5 sweep (APR lr=8, ordinary GD lr=1e-4). No-gate and the
# inverted control are omitted -> just APR vs ordinary GD (+ the merge reference).
source slurm/common.sh
python scripts/compare_baselines.py --config configs/clip8.yaml \
  --steps 15 --clip_mode vdist --skip_inverted \
  --apr_lrs 8 \
  --nogate_lrs \
  --gd_lrs 1e-4 \
  --out results/compare/clip8_s15.json "$@"
