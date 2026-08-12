#!/bin/bash
#SBATCH --job-name=apr-cal-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/calib_clip8-%j.out
#SBATCH --error=logs/calib_clip8-%j.out

# Calibrate-then-refine on CLIP-8: supervised per-tensor coefficient fit on the 64 labeled
# replay examples (the confident-wrong-free analogue of AdaMerging), then APR. Target to
# beat is APR<-AdaMerging = 0.722, but fully inductive (no test-set access). The l2 grid is
# the overfitting guard: ~1600 coefficients on 64 examples/task can memorize, so we sweep
# l2 toward the init and early-stop on a held-out slice of the buffer.
source slurm/common.sh
python scripts/calibrate_refine.py --config configs/clip8.yaml \
  --n_probe 64 --variants layer task --cal_lrs 1e-2 3e-2 --l2_regs 1e-3 1e-2 1e-1 \
  --cal_steps 300 --cal_bs 16 --holdout_frac 0.25 \
  --apr_lrs 4 6 8 --steps 5 \
  --out results/compare/calibrate_refine_clip8.json "$@"
