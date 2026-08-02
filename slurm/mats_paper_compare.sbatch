#!/bin/bash
#SBATCH --job-name=mats-paper-grid
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mats_paper_compare-%j.out
#SBATCH --error=logs/mats_paper_compare-%j.out

set -euo pipefail
cd /home/menglberger/ModelMerging
source slurm/common.sh
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

python scripts/mats_paper_compare.py \
  --config configs/mats_t5_8.yaml \
  --out results/mats_t5_8/paper_compare.json \
  "$@"
