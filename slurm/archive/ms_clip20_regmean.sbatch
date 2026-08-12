#!/bin/bash
#SBATCH --job-name=ms20-regmean
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ms20_regmean-%j.out
#SBATCH --error=logs/ms20_regmean-%j.out

# MULTI-SEED replication, RegMean init (the seed also re-draws the unlabeled
# Gram inputs, so this measures the whole pipeline's seed variance). Submit
# once per replay seed (SEED=1..4; seed 0 = existing). Pinned:
#   apr S30 lr0.5 cosine (.7640) vs gd S30 1e-4 constant+cosine (.7663/.7667)
set -u
S=${SEED:?set SEED via --export=ALL,SEED=...}
source slurm/common.sh
echo "[ms] probe_seed=${S} regmean"

python scripts/regmean_baseline.py --config configs/clip20.yaml \
  --probe_seed "${S}" \
  --n_probe 64 --gram_ns 64 --nondiag_scales 1.0 \
  --apr_lrs 0.5 --apr_schedules cosine --apr_order random \
  --gd_lrs 1e-4 --gd_steps 30 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 30 --select_by mean_acc \
  --out "results/compare/ms_clip20_regmean_seed${S}.json"
