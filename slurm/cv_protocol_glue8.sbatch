#!/bin/bash
#SBATCH --job-name=cv-glue8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/cv_protocol_glue8-%A_%a.out
#SBATCH --error=logs/cv_protocol_glue8-%A_%a.out
#
# Protocol v2 on multi-head GLUE-8, ONE DRAW PER ARRAY TASK (draw seeds
# 100+task_id, matching the single monolithic run this replaces, which timed
# out after draw 0 of 3 in 12h). Two cost lessons from that attempt are baked
# in: the S grid starts at {5,20,50,100,200} because both APR and GD selected
# the longest horizon and forced two S-extensions, and S-extensions now
# CONTINUE each trajectory from its retained state instead of re-running it
# (refine start_sweep; see tests/test_refine_continuation.py for exactness).
#
# Merge the three single-draw files with scripts/merge_cv_draws.py (or read
# them individually; each carries its own protocol block).

set -euo pipefail
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${BUDGET:=32}"
: "${FOLDS:=2}"
: "${INIT:=pretrained}"
: "${ARMS:=apr nogate gd}"          # which labeled arms to run
: "${MAX_STEPS:=apr=100}"           # horizon ceiling tokens (arm=value or int)
: "${SUFFIX:=}"                     # tag for partial reruns, e.g. _aprS100
DRAW=${SLURM_ARRAY_TASK_ID}
SEED=$((100 + DRAW))

OUT="results/compare/cv_glue8_B${BUDGET}_${INIT}_draw${DRAW}${SUFFIX}.json"
WIN="results/cv_winners/glue8_B${BUDGET}_${INIT}/draw${DRAW}${SUFFIX}"

echo "[job] GLUE-8 protocol v2: draw ${DRAW} (seed ${SEED}) B=${BUDGET} init=${INIT}" \
     "arms=[${ARMS}] max_steps=[${MAX_STEPS}] -> ${OUT}"

# shellcheck disable=SC2086  # ARMS / MAX_STEPS are word lists on purpose
python scripts/cv_protocol.py \
  --config configs/glue8.yaml \
  --budget "${BUDGET}" --folds "${FOLDS}" --draws 1 --draw_seed0 "${SEED}" \
  --init "${INIT}" \
  --arms ${ARMS} \
  --apr_lrs 0.5 1 2 4 8 16 32 \
  --nogate_lrs 0.25 0.5 1 2 4 8 \
  --gd_lrs 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 \
  --steps 5 20 50 100 200 \
  --max_steps ${MAX_STEPS} \
  --order random \
  --max_extensions 2 \
  --save_winners "${WIN}" \
  --out "${OUT}"

echo "[done] -> ${OUT}"
