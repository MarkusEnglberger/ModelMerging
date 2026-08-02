#!/bin/bash
#SBATCH --job-name=mats-labeled-base
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mats_labeled_baselines-%j.out
#SBATCH --error=logs/mats_labeled_baselines-%j.out

set -euo pipefail
cd /home/menglberger/ModelMerging
source slurm/common.sh
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

# The shared-output MaTS model has no task-specific classifier to refit, and
# learned full masks are not competitive with the paper's magnitude variant
# (covered in mats_paper_compare). This job runs the remaining matched-label
# baselines selected strictly by replay loss.
python scripts/replay_baselines.py \
  --config configs/mats_t5_8.yaml \
  --n_probe 64 \
  --lam_grid 0.05 0.1 0.2 0.3 0.4 0.5 \
  --skip head ls gd nogate apr \
  --out results/mats_t5_8/labeled_baselines.json \
  "$@"
