#!/bin/bash
#SBATCH --job-name=apr-cal-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/calib_glue8-%j.out
#SBATCH --error=logs/calib_glue8-%j.out

# Calibrate-then-refine on GLUE-8: supervised per-tensor coefficient fit on the 64 labeled
# replay examples (the confident-wrong-free analogue of AdaMerging), then APR. Target to
# beat is the calibrated init working where AdaMerging failed (GLUE 0.173), but fully inductive (no test-set access). The l2 grid is
# the overfitting guard: ~1600 coefficients on 64 examples/task can memorize, so we sweep
# l2 toward the init and early-stop on a held-out slice of the buffer.
source slurm/common.sh
python scripts/calibrate_refine.py --config configs/glue8.yaml \
  --n_probe 64 --variants layer task --cal_lrs 1e-2 3e-2 --l2_regs 1e-3 1e-2 1e-1 \
  --cal_steps 300 --cal_bs 16 --holdout_frac 0.25 \
  --apr_lrs 4 8 16 --steps 5 \
  --out results/compare/calibrate_refine_glue8.json "$@"
