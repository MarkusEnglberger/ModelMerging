#!/bin/bash
#SBATCH --job-name=apr-cal-clip20
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/cal_clip20-%j.out
#SBATCH --error=logs/cal_clip20-%j.out

# Calibrate-then-refine on the 20-task CLIP suite: fit per-tensor scaling
# coefficients by supervised CE (+L2, early-stop on a 25% holdout), then APR.
# This is the principled answer to the lambda caveat AND the CLIP-8 headline
# (0.825): calibration learns the full per-tensor mixing -- a generalized,
# fine-tuned lambda -- so nothing is hand-picked for 20 tasks. l2=1e-1 dropped
# (never won on CLIP-8); apr_lrs extended LOW (2) for the 20-task regime.
# Requires: prefetch_clip.py --suite 20.
source slurm/common.sh
python scripts/calibrate_refine.py --config configs/clip20.yaml \
  --n_probe 64 --variants layer task --cal_lrs 1e-2 3e-2 --l2_regs 1e-3 1e-2 \
  --cal_steps 300 --cal_bs 16 --holdout_frac 0.25 \
  --apr_lrs 2 4 8 --steps 5 --select_by mean_acc \
  --out results/compare/calibrate_refine_clip20_v2.json "$@"
