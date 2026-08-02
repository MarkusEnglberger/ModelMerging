#!/bin/bash
#SBATCH --job-name=mats-fair
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mats_fair_compare-%j.out
#SBATCH --error=logs/mats_fair_compare-%j.out

set -euo pipefail
cd /home/menglberger/ModelMerging
source slurm/common.sh
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

python scripts/mats_fair_compare.py \
  --config configs/mats_t5_8.yaml \
  --n_probe 64 \
  --n_select 64 \
  --apr_lrs 8 16 32 \
  --apr_steps 5 20 \
  --gd_lrs 0.03 0.1 0.3 1.0 \
  --gd_steps 1 5 20 \
  --out results/mats_t5_8/fair_compare.json \
  "$@"
