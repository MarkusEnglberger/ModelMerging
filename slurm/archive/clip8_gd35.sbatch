#!/bin/bash
#SBATCH --job-name=apr-clip8gd35
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/clip8_gd35-%j.out
#SBATCH --error=logs/clip8_gd35-%j.out

# Long-horizon ordinary-GD-only run on the 8-task CLIP suite: S=35 sweeps at the
# best GD learning rate (1e-4), to see whether GD converges / catches up given many
# more steps. No APR/no-gate/inverted (empty grids + --skip_inverted).
source slurm/common.sh
python scripts/compare_baselines.py --config configs/clip8.yaml \
  --steps 35 --clip_mode vdist_pre --gammas 1.0 --skip_inverted \
  --apr_lrs \
  --nogate_lrs \
  --gd_lrs 1e-4 \
  --out results/compare/clip8_gd35.json "$@"
