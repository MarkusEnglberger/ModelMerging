#!/bin/bash
#SBATCH --job-name=mb3b-gdmx
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mb3b_gdmx-%j.out
#SBATCH --error=logs/mb3b_gdmx-%j.out

# The decisive regime-map test at 3B (~1.2e7 params per replay example):
# gated APR vs ungated anchored vs ordinary GD, all from the SAME tuned
# TA@l0.5 init (pinned: the selected winner of the composition job's sweep;
# apr@lr64 repeated as the cross-job anchor cell).
#
# Registered predictions (before results): GD's best cell below both anchored
# arms, worst-task gap larger; a catastrophic GD cell at the top of its grid;
# ungated >= gated on mean.
#
# PRIOR EVIDENCE AGAINST prediction 1, recorded for honesty: results/compare/
# mergebench3b_compare{,_v2}.json both had GD >= APR. They are not decisive --
# 3 tasks, untuned lambda=0.3, an eval protocol still being repaired (the two
# disagree on eval0 for the SAME merge), and APR capped at lr16 when its optimum
# later proved to be lr64 -- but they are why GD's grid reaches below 1e-5:
# GD peaked at 1e-5/5e-5 there, so a grid starting higher would edge-clip the
# arm this experiment exists to test fairly.
#
# COST CONTROL (this is a REDUCED first pass; see the two TODOs below).
# The first attempt at this job (25178513) was cancelled 2h26m in: it needed 17
# evaluation passes at ~67 min each (~19-21 h) against a 16 h limit, and the
# report is only written at the end, so it would have produced nothing. The cost
# is dominated by IFEval generation (2853 s/eval, ~72% of every pass). Two
# reductions bring it to ~10 h with margin:
#   1. configs/mergebench_llama3b_sweep.yaml halves instruction eval (300->150).
#   2. The grid is trimmed 13 -> 9 cells. GD keeps the widest coverage of the
#      three arms because it is the arm under test; the anchored arms are cut
#      harder, which biases AGAINST our own method rather than for it.
#
# TODO(extend): after this lands, re-run selected cells with the full
# configs/mergebench_llama3b.yaml (instruction=300) before any number enters the
# paper -- the 150-item instruction scores are not comparable to the 300-item
# runs already on disk.
# TODO(extend): restore the dropped cells (apr 256 is kept; nogate 16/64/256 and
# gd 3e-6/3e-5/3e-4 were dropped) once the coarse picture is known.

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b_sweep.yaml \
  --ta_lams 0.5 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --apr_lrs 64 128 256 \
  --nogate_lrs 32 128 \
  --control_gd_lrs 1e-5 1e-4 3e-4 1e-3 \
  --out results/compare/mb3b_gd_matrix_sweep.json
