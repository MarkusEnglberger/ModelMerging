#!/bin/bash
#SBATCH --job-name=apr-rgm-val32
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=logs/regmean_val3232-%j.out
#SBATCH --error=logs/regmean_val3232-%j.out

# LOOSE END 2 / PROTOCOL PILOT: honest hyperparameter selection WITHOUT the
# test set, 32+32 budget split (user decision): the historical 64 labeled
# samples/task are split disjointly into 32 sweep samples + 32 held-out
# validation samples (union == the old 64-buffer, verified). Every cell records
# val_acc; selection can then be done on val and reported on test, and the
# val-vs-oracle selection gap measured. RegMean init (the headline non-calib
# init): grams also restricted to the 32 train samples (honest budget), APR and
# GD arms at S=30 constant+cosine. No nogate arms (user decision).
set -euo pipefail
cd /home/menglberger/ModelMerging
source slurm/common.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/regmean_baseline.py --config configs/clip20.yaml \
  --n_probe 32 --n_val 32 --gram_ns 32 --gram_bs 32 --nondiag_scales 1.0 \
  --apr_lrs 0.25 0.5 1 \
  --apr_schedules constant cosine --apr_order random \
  --gd_lrs 5e-5 1e-4 --gd_steps 30 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 30 --select_by mean_acc \
  --out results/compare/regmean_clip20_val3232.json "$@"
