#!/bin/bash
# Payload of the budget-matched t2t S<=200 column: one merge_baselines.py run.
# Called by t5_matched_s200.sbatch (one run per job) and
# t5_matched_s200_paired.sbatch (two runs co-scheduled on one GPU).
# See t5_matched_s200.sbatch for the rationale behind the grids.

set -euo pipefail

: "${MODE:?set MODE (merges|base)}"
: "${INITS:=ta ties dareties bc}"
: "${SEED:=0}"

TAG=$(echo "$INITS" | tr -d ' ')
OUT="results/compare/grid_nn_t5_glue8_${MODE}_n16_s200${TAG:+_$TAG}.json"

GD="3e-5 1e-4 3e-4 1e-3 3e-3 1e-2"
if [ "$MODE" = base ]; then
  APR="16 32 64 128 256 512 1024"
  NG="2 4 8 16 32 64 128"
else
  APR="1 2 4 8 16 32"
  NG="0.125 0.25 0.5 1 2 4 8"
fi

COMMON=(--config configs/t5_glue8.yaml
        --n_probe 16 --n_select 16
        --probe_seed "$SEED"
        --steps 5 20 50 100 200
        --select_by mean_normret
        --apr_lrs $APR
        --nogate_lrs $NG
        --control_gd_lrs $GD
        --out "$OUT")

echo "[job] t2t matched S<=200 MODE=$MODE INITS='$INITS' seed=$SEED -> $OUT"

if [ "$MODE" = base ]; then
  # --ta_lams 0 makes merge:TA@l0 identical to theta_0, so refining "from ta"
  # refines from the PRETRAINED model.
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --ta_lams 0 \
    --skip_families ties dare dareties della bc ls ada \
    --refine_from ta
else
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --skip_families dare della ls ada \
    --refine_from $INITS
fi

echo "[done] -> $OUT"
