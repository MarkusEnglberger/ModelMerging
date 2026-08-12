#!/bin/bash
#SBATCH --job-name=t5-s200
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --output=logs/t5_matched_s200-%j.out
#SBATCH --error=logs/t5_matched_s200-%j.out
#
# A properly bracketed, BUDGET-MATCHED shared-output (t2t) column at n=16+16,
# mirroring slurm/clip8_matched_s200.sbatch. Same protocol, no special
# treatment: all three arms get the same horizon grid S in {5,20,50,100,200}
# and lr grids re-centred so the previously selected cells are interior.
#
#   sbatch --export=ALL,MODE=base                            slurm/t5_matched_s200.sbatch
#   sbatch --export=ALL,MODE=merges,INITS="ta ties"          slurm/t5_matched_s200.sbatch
#   sbatch --export=ALL,MODE=merges,INITS="dareties bc"      slurm/t5_matched_s200.sbatch
#
# Why this run exists: on CLIP-8 the "GD wins the composition rows" ordering
# was a search-budget artifact -- APR was edge-clipped (its optimum needed a
# lower lr and a longer horizon along the lr*S ridge) while GD's cells were
# interior; the budget-matched column flipped TIES and DARE-TIES to APR. The
# t2t column has exactly the same asymmetry today: APR's selected cell sits at
# the lr grid MINIMUM (32) in all four composition rows while GD's 0.716
# headline is interior. Whether "GD wins the shared-output track outright"
# survives a bracketed search is the question; the run answers it in either
# direction.
#
# Grids (re-centred, not just widened -- the ridge means a longer horizon
# needs a proportionally lower rate):
#   APR merges: old picks all lr=32 at the grid MIN -> extend DOWN to 1
#               (ridge from (32, S=5) reaches ~1 at S=200).
#   APR base:   the from-pretrained regime lives at much higher rates
#               (old pick lr=512 at S=50 -> ~128 at S=200); grid stays high.
#   NG merges:  old picks lr=4 at MIN -> extend down to 0.125.
#   NG base:    old pick lr=32 at MAX -> extend up to 128.
#   GD both:    old picks 3e-4..3e-3, one at MIN -> extend down to 3e-5.
#
# AdaMerging stays excluded on this track (its entropy objective needs a
# classification head; documented capability gap, not an unrun experiment).
#
# T5 sweeps cost ~10s each (vs ~3s on CLIP), so: base ~20 trajectories x 200
# sweeps ~ 12h; each merges pair ~38 x 200 ~ 22h + merge-cell evals. 100G
# keeps the job inside a single GPU billing share.

set -euo pipefail
source slurm/common.sh

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
