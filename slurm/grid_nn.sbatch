#!/bin/bash
#SBATCH --job-name=grid-nn
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --output=logs/grid_nn-%j.out
#SBATCH --error=logs/grid_nn-%j.out
#
# The strict, deployment-facing tier of the main experiment grid: the n+n
# split. Every hyperparameter -- merge coefficient, density, refinement
# learning rate AND sweep count -- is chosen on a labeled selection buffer that
# is disjoint from the replay buffer, so the evaluation split is touched only
# to score the finally selected cells. Total labeled cost is stated honestly as
# n_probe + n_select per task.
#
#   sbatch --export=ALL,BENCH=clip8,MODE=merges,N=64 slurm/grid_nn.sbatch
#   sbatch --export=ALL,BENCH=clip20,MODE=base,N=16  slurm/grid_nn.sbatch
#
# Two protocol points this job exists to get right:
#
#   1. The sweep count S is part of the grid, not pinned. The (lr, S) optima
#      trade off along a roughly constant lr*S product, and the arms converge at
#      different horizons -- APR near-converges by S=5-10 while ordinary GD is
#      still improving at S=35 -- so fixing one S is not neutral between them
#      and, at S=5, is biased in our own method's favour.
#   2. Selection never sees the evaluation split. Each cell also records its
#      selection objective, so the split-selected and evaluation-selected
#      answers can be reported side by side; their difference is the honest
#      selection gap of a procedure a practitioner could actually run.
#
# What this does NOT establish: a cell selected here is not certified safe. The
# hazard is a property of the replay draw, not of the cell -- descent's selected
# optimum has diverged under nothing more than a buffer redraw -- so multi-seed
# replication remains the only instrument for that claim.

set -euo pipefail
source slurm/common.sh

: "${BENCH:?set BENCH (glue8|clip8|clip20|t5_glue8)}"
: "${MODE:?set MODE (merges|base)}"
: "${N:=64}"

# AdaMerging is excluded on the shared-output track: its entropy objective
# assumes a classification head and the graph detaches on a seq2seq LM head.
case "$BENCH" in
  glue8)     CFG=configs/glue8.yaml;       APR="1 2 4 8 16 32";        NG="0.5 1 2 4 8";  GD="1e-5 5e-5 1e-4 5e-4 1e-3"; INITS="ta ties dareties bc ada"; SKIP="dare della ls" ;;
  clip8)     CFG=configs/clip8.yaml;       APR="2 4 8 16 32";          NG="2 4 8 16";     GD="1e-4 5e-4 1e-3 5e-3";      INITS="ta ties dareties bc ada"; SKIP="dare della ls" ;;
  clip20)    CFG=configs/clip20_l008.yaml; APR="1 2 4 8 16";           NG="1 2 4 8";      GD="1e-5 1e-4 5e-4 1e-3";      INITS="ta ties dareties bc ada"; SKIP="dare della ls" ;;
  t5_glue8)  CFG=configs/t5_glue8.yaml;    APR="64 128 256 512 1024";  NG="4 8 16 32";    GD="1e-3 3e-3 1e-2 3e-2";      INITS="ta ties dareties bc";     SKIP="dare della ls ada" ;;
  *) echo "unknown BENCH=$BENCH" >&2; exit 2 ;;
esac

# Overrides for extension waves. The (lr, S) optima trade off along a roughly
# constant lr*S product, so a longer horizon needs a PROPORTIONALLY LOWER
# learning-rate grid: going from S=20 to S=50 (2.5x) shifts the grid down by
# about the same factor. Extending S while holding lr fixed would test the
# wrong cells.
: "${STEPS_GRID:=5 20}"
[ -n "${APR_OVR:-}" ] && APR="$APR_OVR"
[ -n "${NG_OVR:-}"  ] && NG="$NG_OVR"
[ -n "${GD_OVR:-}"  ] && GD="$GD_OVR"
[ -n "${TAG:-}" ] && SUFFIX="_$TAG" || SUFFIX=""

SELECT=mean_normret
[ "$BENCH" = clip20 ] && SELECT=mean_acc

OUT="results/compare/grid_nn_${BENCH}_${MODE}_n${N}${SUFFIX}.json"
# AdaMerging's unlabeled inputs come from the MATCHED BUDGET, not the eval
# split. Its objective minimises prediction entropy on unlabeled data, and the
# standard formulation draws that data from the evaluation set -- transductive:
# it adapts to the very test points it is scored on (87,433 images on CLIP-8,
# against the 512 labeled examples APR gets). The paper reports the matched-
# budget variant, which sees the same n unlabeled inputs per task that APR sees
# labeled, so this flag is required for the rows to be comparable.
COMMON=(--config "$CFG" --n_probe "$N" --n_select "$N" --steps $STEPS_GRID
        --select_by "$SELECT" --apr_lrs $APR --nogate_lrs $NG
        --control_gd_lrs $GD --ada_data probe_buffer --out "$OUT")

echo "[job] BENCH=$BENCH MODE=$MODE n=${N}+${N} S in {$STEPS_GRID} -> $OUT"

if [ "$MODE" = base ]; then
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --ta_lams 0 \
    --skip_families ties dare dareties della bc ls ada \
    --refine_from ta
else
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --skip_families $SKIP \
    --refine_from $INITS
fi
