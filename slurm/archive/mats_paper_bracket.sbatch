#!/bin/bash
# Bracket the grids behind the paper's headline shared-output T5 comparison.
#
# results/mats_t5_8/paper_compare.json selected a boundary cell on three axes:
#   APR short   lr=32   = max of [2,4,8,16,32]
#   APR long    lr=32   = max of [4,8,16,32]
#   ordinary GD lr=0.3  = max of [1e-4 .. 0.3]
#   TA          lam=0.2 = min of [0.2,0.3,0.4,0.5]
# so none of those numbers is a tuned optimum. The grids below open every one of
# those boundaries while keeping enough of the old range to confirm the previous
# winner is reproduced. The ungated axis already selected in the interior
# (lr=32 of [0.5..64]) and is widened only to stay comparable.
#
#SBATCH --job-name=mats-pbrkt
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mats_paper_bracket-%j.out
#SBATCH --error=logs/mats_paper_bracket-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/common.sh
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

python scripts/mats_paper_compare.py \
  --config configs/mats_t5_8.yaml \
  --ta_lams 0.05 0.1 0.2 0.3 0.4 \
  --apr_lrs 8 16 32 64 128 256 \
  --nogate_lrs 8 16 32 64 128 256 \
  --gd_lrs 0.03 0.1 0.3 1.0 3.0 10.0 \
  --short_steps 5 \
  --long_steps 20 \
  --long_lrs 16 32 64 128 256 \
  --long_min_fracs 0.05 0.2 \
  --out results/mats_t5_8/paper_compare_bracket.json
