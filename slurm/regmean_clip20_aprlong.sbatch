#!/bin/bash
#SBATCH --job-name=apr-rgm-long
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/regmean_aprlong-%j.out
#SBATCH --error=logs/regmean_aprlong-%j.out

# LOOSE END 1: the missing APR-long-from-RegMean fairness cell. gd<-regmean
# @1e-4 S60cosine (.7677/.5106) is the best non-calib cell in the project but
# apr<-regmean only exists at S=5 (.7485). apr30 showed APR was NOT converged
# at S5 (ties/ada gained +.005-.011 mean and +.15 worst from S=30), so APR
# deserves the same budget here. Old n=64 protocol ON PURPOSE: these cells must
# be comparable with the existing gd<-regmean S30/S60 cells (same buffers).
# lr grid around the S5 optimum 0.5; constant + cosine/random, S=30.
source slurm/common.sh
python scripts/regmean_baseline.py --config configs/clip20.yaml \
  --n_probe 64 --gram_ns 64 --nondiag_scales 1.0 \
  --apr_lrs 0.25 0.5 1 \
  --apr_schedules constant cosine --apr_order random \
  --gd_lr_min_frac 0.05 \
  --steps 30 --select_by mean_acc \
  --out results/compare/regmean_clip20_aprlong.json "$@"
