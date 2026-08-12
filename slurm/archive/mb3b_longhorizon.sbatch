#!/bin/bash
#SBATCH --job-name=mb3b-S20
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mb3b_S20-%j.out
#SBATCH --error=logs/mb3b_S20-%j.out

# Tests the "GD just needs more sweeps" objection at 3B -- the objection our own
# S=5 result raises. The reduced sweep found GD's arc FLAT (0.448-0.458 across
# 1e-5..1e-3) rather than divergent, i.e. plain descent at 5 sweeps x 64 examples
# on 3.2e9 parameters is under-powered, not unstable. If that diagnosis is right,
# GD should improve with a longer horizon, and the S=5 comparison alone cannot
# support "the anchored arms beat descent at 3B".
#
# All three arms run at S=20 (4x the horizon) so the comparison stays matched;
# the S=5 numbers come from mb3b_gd_matrix_sweep.json at the same protocol.
# GD keeps 3 lrs against 1 for each anchored arm, as before.
#
# Cheap because evaluation dominates and refinement does not: an S=20 cell costs
# ~16 min of refine against ~57 min of eval, so quadrupling the horizon adds ~20%
# per cell rather than 4x. eval0 is already cached at this protocol
# (results/eval0_cache/7d6708d306eff1b2.json), so it costs 0 passes.
#
# COST: 1 pass (TA@l0.5) + 5 cells = 6 passes x ~57 min + ~80 min refine ~= 7 h
# against a 12 h limit.
#
# NOTE this runs at the SWEEP protocol (instruction=150), so it answers the
# directional question -- does more compute close the gap? -- and is comparable
# to mb3b_gd_matrix_sweep.json, NOT to the 300-item runs. If GD closes the gap
# here, the winning cells must be re-scored at instruction=300 before use.

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b_sweep.yaml \
  --ta_lams 0.5 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --steps 20 \
  --apr_lrs 256 \
  --nogate_lrs 128 \
  --control_gd_lrs 1e-5 1e-4 1e-3 \
  --out results/compare/mb3b_longhorizon_S20.json
