#!/bin/bash
#SBATCH --job-name=nn-smoke
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nn_smoke-%j.out
#SBATCH --error=logs/nn_smoke-%j.out
#
# Validates the n+n split wiring in merge_baselines.py before the real runs:
# that the selection buffer is drawn disjointly, that selection_obj is recorded
# on every cell (merge families and all three refinement arms), and that
# selection no longer touches the evaluation split. Tiny grids on purpose.

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config configs/glue8.yaml \
  --n_probe 8 --n_select 8 --steps 2 4 \
  --ta_lams 0.2 0.3 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --apr_lrs 4 8 \
  --nogate_lrs 2 \
  --control_gd_lrs 1e-4 \
  --out results/compare/nn_split_smoke.json
