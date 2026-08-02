#!/bin/bash
# Cheap end-to-end check of the bracketing job's code path: one family, one
# learning rate, 1-sweep horizons, tiny buffers. Verifies --probe_seed plumbing,
# the disjointness assertion and the report schema before the real jobs run.
#SBATCH --job-name=mats-smoke
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/mats_bracket_smoke-%j.out
#SBATCH --error=logs/mats_bracket_smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/common.sh
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

python scripts/mats_fair_compare.py \
  --config configs/mats_t5_8.yaml \
  --n_probe 8 \
  --n_select 8 \
  --probe_seed 2 \
  --selection_seed 3 \
  --grad_batch_size 8 \
  --apr_lrs 32 \
  --apr_steps 1 \
  --gd_lrs 1 \
  --gd_steps 1 \
  --families TIES \
  --out results/mats_t5_8/bracket_smoke.json
