#!/bin/bash
#SBATCH --job-name=gdmx-smoke
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/gdmx_smoke-%j.out
#SBATCH --error=logs/gdmx_smoke-%j.out

# Smoke test for the new --gd_lrs/--gd_steps/--gd_schedules/--nogate_lrs arms
# added to merge_baselines.py. 3-task POC with tiny grids: proves the code path
# runs end-to-end and records apr/nogate/gd cells before ~30 GPU-hours of the
# real init x method matrix are committed to it.
# (Must run on a GPU node -- the login node has no GPU and configs set device:cuda.)
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip_poc.yaml \
  --ta_lams 0.3 --skip_families ties dare dareties bc ada \
  --refine_from ta --apr_lrs 8 --nogate_lrs 1 \
  --gd_lrs 1e-4 --gd_steps 2 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random --steps 2 --select_by mean_acc \
  --out results/compare/_gdmx_smoke.json
