#!/bin/bash
#SBATCH --job-name=c8-s200
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --output=logs/clip8_matched_s200-%j.out
#SBATCH --error=logs/clip8_matched_s200-%j.out
#
# A properly bracketed, BUDGET-MATCHED CLIP-8 column at n=16+16.
#
#   sbatch --export=ALL,MODE=base                        slurm/clip8_matched_s200.sbatch
#   sbatch --export=ALL,MODE=merges,INITS="ta ties dareties" slurm/clip8_matched_s200.sbatch
#   sbatch --export=ALL,MODE=merges,INITS="bc ada"           slurm/clip8_matched_s200.sbatch
#
# Two defects in the S<=50 column this fixes, both found by the schedule study:
#
#   1. IT WAS EDGE-CLIPPED. Extending the horizon to S=100 lifted constant-rate
#      APR from TIES by 0.061 at the same replay draw (0.548 -> 0.609), and the
#      selected horizon still sat at the grid maximum in 16 of 18 cells. From a
#      merge initialization, APR at n=16 wants a longer refinement than either
#      grid so far offered, so this one goes to S=200.
#   2. THE ARMS WERE NOT BUDGET-MATCHED. The schedule study searched S<=100 for
#      APR while the GD and ungated cells beside it had only seen S<=50. All
#      three arms here get the SAME horizon grid, so the column compares
#      methods rather than search budgets.
#
# The annealed arm is deliberately dropped. Across 18 paired comparisons it was
# worth +0.002 +- 0.032 and won 9 of 18 -- indistinguishable from zero -- so
# re-running it would buy nothing and every arm here uses the constant rate and
# fixed task order that the paper uses throughout.
#
# All three arms use a constant schedule, so a single trajectory per learning
# rate yields every horizon by snapshotting: cost is n_lr * max(S) per arm, not
# n_lr * sum(S). That is what makes S=200 affordable at all.
#
# Learning-rate grids are re-centred DOWNWARD for the longer horizon. The (lr,S)
# optima trade off along a roughly constant lr*S product, so quadrupling the
# horizon from the S<=50 grids divides the matched rate by about four; each grid
# below is centred on that prediction with at least two octaves of headroom on
# each side, since edge-clipping is the exact failure being repaired.
#
#   APR      S<=100 optima were lr 4-8 from merges, 32 from pretrained -> 1..32
#   ungated  S<=50  optimum  was lr 2                                  -> 0.125..4
#   GD       S<=50  optimum  was lr 1e-4                               -> 1e-6..5e-4
#
# Split across jobs by initialization to stay inside the 20h wall clock: at
# ~0.06 min/sweep, three arms x 6 rates x 200 sweeps is ~3.6k sweeps per
# initialization, about 3.6h.

set -euo pipefail
source slurm/common.sh

: "${MODE:?set MODE (merges|base)}"
: "${INITS:=ta ties dareties bc ada}"
: "${SEED:=0}"

TAG=$(echo "$INITS" | tr -d ' ')
OUT="results/compare/grid_nn_clip8_${MODE}_n16_s200${TAG:+_$TAG}.json"

COMMON=(--config configs/clip8.yaml
        --n_probe 16 --n_select 16
        --probe_seed "$SEED"
        --steps 5 20 50 100 200
        --select_by mean_normret
        --apr_lrs 1 2 4 8 16 32
        --nogate_lrs 0.125 0.25 0.5 1 2 4
        --control_gd_lrs 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
        --ada_data probe_buffer
        --out "$OUT")

echo "[job] CLIP-8 matched S<=200 MODE=$MODE INITS='$INITS' seed=$SEED -> $OUT"

if [ "$MODE" = base ]; then
  # --ta_lams 0 makes merge:TA@l0 identical to theta_0, so refining "from ta"
  # refines from the PRETRAINED model.
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --ta_lams 0 \
    --skip_families ties dare dareties della bc ls ada \
    --refine_from ta
else
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --skip_families dare della ls \
    --refine_from $INITS
fi

echo "[done] -> $OUT"
