#!/bin/bash
#SBATCH --job-name=cv-clip8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=10:00:00
# 10h, not 20: the identical B=32 / 3-draw run took 7h08m, and walltime
# does not enter the priority formula but does gate backfill -- a 20h
# request only fits a >=20h gap, nearly unschedulable while the partition
# is saturated on GPUs.
#SBATCH --output=logs/cv_protocol_clip8-%j.out
#SBATCH --error=logs/cv_protocol_clip8-%j.out
#
# Protocol v2 on the eight-task CLIP suite: equal total label budget (B per
# task for EVERY method), K-fold select-then-refit for the labeled arms,
# D independent buffer draws as the unit of reporting, grids auto-extended
# until the selected cell is interior, per-fold loss traces and refit winners
# persisted.  See scripts/cv_protocol.py for the rationale of each element.
#
# Budget: B=32 matches the total label spend of the n=16+16 split-selection
# rows in Table 1, so the two are budget-comparable -- but note the labeled
# arms now CONSTRUCT on up to 32 examples (24 per CV fold, 32 at refit)
# instead of 16, so this is a protocol change, not a re-run.
#
# The merge grids are the same candidate sets the split-selection runs used,
# now scored on all B examples rather than on the n selection examples.

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"      # KEY=VALUE positional args; env does NOT propagate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${BUDGET:=32}"
: "${FOLDS:=2}"
: "${DRAWS:=3}"
: "${INIT:=pretrained}"
: "${ARMS:=apr nogate gd}"
: "${SUFFIX:=}"

OUT="results/compare/cv_clip8_B${BUDGET}_${INIT}${SUFFIX}.json"
WIN="results/cv_winners/clip8_B${BUDGET}_${INIT}${SUFFIX}"

echo "[job] CLIP-8 protocol v2: B=${BUDGET} folds=${FOLDS} draws=${DRAWS} init=${INIT}" \
     "arms=[${ARMS}] -> ${OUT}"
guard_output "${OUT}"

# shellcheck disable=SC2086  # ARMS is a word list on purpose
python scripts/cv_protocol.py \
  --config configs/clip8.yaml \
  --budget "${BUDGET}" --folds "${FOLDS}" --draws "${DRAWS}" \
  --init "${INIT}" \
  --arms ${ARMS} \
  --apr_lrs 1 2 4 8 16 32 \
  --nogate_lrs 0.5 1 2 4 8 16 \
  --gd_lrs 1e-5 1e-4 5e-4 1e-3 5e-3 1e-2 \
  --steps 5 20 50 100 \
  --order random \
  --max_extensions 2 \
  --save_winners "${WIN}" \
  --out "${OUT}"

echo "[done] -> ${OUT}"
