#!/bin/bash
#SBATCH --job-name=t5-brkt
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/t5_bracket-%j.out
#SBATCH --error=logs/t5_bracket-%j.out
#
# Three-arm bracketing on the full-model shared-head T5-GLUE track.
#
# WHY THIS TRACK: it is the missing middle of the parameters-per-example regime
# map, and it is ~13x cheaper per evaluation pass than the 3B decoder track
# (~5 min vs ~60-74 min) because scoring is short text-to-text rather than
# generation. flan-T5-base experts are genuine FULL fine-tunes (247.6M params,
# 282 tensors, no PEFT artifacts) sharing theta_0 = google/flan-t5-base, and the
# whole seq2seq model including lm_head is merged.
#
#   MaTS-IA3    282k adapter params   ~550 params/example    GD wins
#   T5-GLUE     247.6M full params    ~4.8e5                 <- THIS JOB
#   mb3b        3.2B full params      ~1.2e7                 anchored wins
#
# Same architecture family and shared generation head as MaTS, ~900x the merged
# parameters, so it isolates underdetermination from modality and head-sharing.
#
# WHAT IS BROKEN TODAY: every arm on this track is edge-clipped. In
# merge_baselines_t5_glue8.json, APR peaks at lr32 -- the grid maximum -- from
# ALL FOUR inits tried (ta/ties/dareties/bc), and that job has no ungated or GD
# arms at all. In the older t5_glue8_compare.json all three arms peak at their
# grid maxima (apr lr32, nogate lr8, gd 1e-3). Nothing on this track is
# bracketed, and no horizon other than S=5 has ever been run.
#
# THIS JOB extends every grid upward past its old edge, from the tuned TA@l0.3
# init (the interior lambda optimum reported in the paper). S is set by $STEPS so
# the same script covers both horizons:
#
#   sbatch --export=ALL,STEPS=5  slurm/t5_glue8_bracket.sbatch
#   sbatch --export=ALL,STEPS=20 slurm/t5_glue8_bracket.sbatch
#
# COST: eval0 (3) + TA@l0.3 (1) + 12 cells = 16 passes at ~5.4 min = ~1.5 h at
# S=5, ~2.5 h at S=20 (refinement is 4x but eval still dominates). Sized from
# the July job's measured 38 methods in 3h25m.
#
# SELECTION: merge_baselines.py is evaluation-selected (--select_by), matching
# this track's existing protocol; replay_obj is recorded per cell so the
# selection gap can be measured post hoc, as it was on MaTS.

set -euo pipefail
source slurm/common.sh

: "${STEPS:=5}"
OUT="results/compare/t5_glue8_bracket_S${STEPS}.json"
echo "[job] three-arm bracketing, S=${STEPS} -> ${OUT}"

python scripts/merge_baselines.py \
  --config configs/t5_glue8.yaml \
  --ta_lams 0.3 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --steps "${STEPS}" \
  --apr_lrs 32 64 128 256 \
  --nogate_lrs 8 16 32 64 \
  --control_gd_lrs 1e-3 3e-3 1e-2 3e-2 \
  --out "${OUT}"
