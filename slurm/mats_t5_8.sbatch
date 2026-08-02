#!/bin/bash
#SBATCH --job-name=apr-mats-ia3
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mats_t5_8-%j.out
#SBATCH --error=logs/mats_t5_8-%j.out

set -euo pipefail
cd /home/menglberger/ModelMerging
source slurm/common.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

python scripts/run_merge.py --config configs/mats_t5_8.yaml "$@"
