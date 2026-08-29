#!/bin/bash
#SBATCH --job-name=cv-v2
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=logs/cv-%x-%A_%a.out
#SBATCH --error=logs/cv-%x-%A_%a.out
#
# Generic protocol-v2 runner: ONE BUFFER DRAW PER ARRAY TASK.
#
# Splitting by draw rather than looping inside one job cuts wall-clock by 3x
# and, because walltime gates backfill without entering the priority formula,
# a 3x6h array schedules far sooner than a single 18h job. Draw d uses buffer
# seed 100+d, matching every earlier v2 run. Stitch the per-draw files with
#   python scripts/merge_cv_draws.py OUT.json OUT_draw0.json OUT_draw1.json ...
#
# Parameters are positional KEY=VALUE arguments (the environment is NOT
# propagated to jobs here -- see slurm/args.sh):
#
#   sbatch --array=0-2 --time=6:00:00 -J clip8-B16 slurm/cv_protocol.sbatch \
#          CONFIG=configs/clip8.yaml TAG=clip8 BUDGET=16
#
#   sbatch --array=0-2 --time=16:00:00 -J clip20-B32 slurm/cv_protocol.sbatch \
#          CONFIG=configs/clip20.yaml TAG=clip20 BUDGET=32 \
#          "TA_LAMS=0.05 0.08 0.1 0.15 0.2 0.3"
#
# INIT selects what the labeled arms refine from; anything other than
# "pretrained" runs APR alone (the GD/ungated ablations are reported only from
# the pretrained model), which cv_protocol.py enforces.

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CONFIG:?CONFIG=configs/<suite>.yaml is required}"
: "${TAG:?TAG=<suite> is required (names the output file)}"
: "${BUDGET:=32}"
: "${FOLDS:=2}"
: "${INIT:=pretrained}"
: "${ARMS:=apr nogate gd}"
: "${STEPS:=5 20 50 100}"
# Uniform horizon ceiling for APR across suites and budgets, so the B=16 and
# B=32 tables are directly comparable. A selection at the ceiling is recorded
# as S_capped rather than triggering another doubling of the grid.
: "${MAX_STEPS:=apr=100}"
: "${APR_LRS:=1 2 4 8 16 32}"
: "${NOGATE_LRS:=0.5 1 2 4 8 16}"
: "${GD_LRS:=1e-5 1e-4 5e-4 1e-3 5e-3 1e-2}"
: "${TA_LAMS:=0.2 0.3 0.4 0.5}"
: "${EXTRA:=}"                 # e.g. EXTRA=--fisher
: "${SUFFIX:=}"

: "${SEED0:=100}"              # buffer seed of array task 0; 103 = fresh draws
DRAW=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((SEED0 + DRAW))

OUT="results/compare/cv_${TAG}_B${BUDGET}_${INIT}${SUFFIX}_draw${DRAW}.json"
WIN="results/cv_winners/${TAG}_B${BUDGET}_${INIT}${SUFFIX}/draw${DRAW}"

echo "[job] protocol v2: ${TAG} B=${BUDGET} init=${INIT} draw=${DRAW} (seed ${SEED})"
echo "      arms=[${ARMS}] steps=[${STEPS}] max_steps=[${MAX_STEPS}] extra=[${EXTRA}]"
echo "      -> ${OUT}"
guard_output "${OUT}"

# shellcheck disable=SC2086  # the grids are word lists on purpose
python scripts/cv_protocol.py \
  --config "${CONFIG}" \
  --budget "${BUDGET}" --folds "${FOLDS}" --draws 1 --draw_seed0 "${SEED}" \
  --init "${INIT}" \
  --arms ${ARMS} \
  --apr_lrs ${APR_LRS} \
  --nogate_lrs ${NOGATE_LRS} \
  --gd_lrs ${GD_LRS} \
  --ta_lams ${TA_LAMS} \
  --steps ${STEPS} \
  --max_steps ${MAX_STEPS} \
  --order random \
  --max_extensions 2 \
  --save_winners "${WIN}" \
  --out "${OUT}" \
  ${EXTRA}

echo "[done] -> ${OUT}"
