#!/bin/bash
#SBATCH --job-name=apr-cal-t5
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/calib_t5_glue8-%j.out
#SBATCH --error=logs/calib_t5_glue8-%j.out

# Calibrate-then-refine on the SHARED-HEAD T5 GLUE-8 track (flan-t5-base, text-to-text):
# Stage 1 fits per-tensor merge coefficients on the 64 labeled replay examples (the
# confident-wrong-free analogue of AdaMerging); Stage 2 runs APR from the calibrated
# merge. Fully inductive -- both stages touch only the replay buffer, never the eval split.
# This is the T5 analogue of calibrate_refine_glue8.sbatch. On T5, AdaMerging was skipped
# (entropy objective needs a seq2seq adaptation), so the calibrated init is the ONLY
# supervised per-tensor merge point here -- it also answers "can a labeled per-tensor fit
# give APR a better launch point than plain TA on the shared-head setting?".
#
# APR lr grid {8,16,32,64}: shifted UP vs the GLUE-8 calibrate run because every T5 APR
# family (compare + merge-baselines) peaked at lr32 (grid edge); we bracket higher so the
# APR-from-calibrated optimum is interior, not pegged. Benefits from the STS-B metrics NaN
# guard, so aggregates are clean. Requires: python scripts/prefetch_t5.py (already cached).
source slurm/common.sh
python scripts/calibrate_refine.py --config configs/t5_glue8.yaml \
  --n_probe 64 --variants layer task --cal_lrs 1e-2 3e-2 --l2_regs 1e-3 1e-2 1e-1 \
  --cal_steps 300 --cal_bs 16 --holdout_frac 0.25 \
  --apr_lrs 8 16 32 64 --steps 5 \
  --out results/compare/calibrate_refine_t5_glue8.json "$@"
