#!/bin/bash
#SBATCH --job-name=della-apr-mats
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/della_apr_mats-%j.out
#SBATCH --error=logs/della_apr_mats-%j.out

set -euo pipefail
cd /home/menglberger/ModelMerging
source slurm/common.sh
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

# Paper-default DELLA followed by the APR grid used by the MaTS fair comparison.
python scripts/merge_baselines.py --config configs/mats_t5_8.yaml \
  --della_densities 0.7 --della_windows 0.14 --della_lams 1.1 --della_seeds 42 \
  --skip_families ta ties dare dareties bc ls ada \
  --refine_from della --apr_lrs 8 16 32 --steps 5 \
  --out results/mats_t5_8/della_apr.json "$@"
