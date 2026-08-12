#!/bin/bash
#SBATCH --job-name=grid-n16
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grid_n16-%j.out
#SBATCH --error=logs/grid_n16-%j.out
#
# Fills the paper's main experiment grid (Table 1) at n=16 replay examples per
# task, so the grid can be reported at two budgets on every benchmark.
#
#   sbatch --export=ALL,BENCH=glue8,MODE=merges slurm/grid_n16.sbatch
#   sbatch --export=ALL,BENCH=clip8,MODE=base   slurm/grid_n16.sbatch
#
# MODE=merges  every merge initialization (TA, TIES, DARE-TIES, Breadcrumbs,
#              AdaMerging) alone and refined. --nogate_lrs/--control_gd_lrs run
#              from each initialization, so this also supplies the ungated and
#              GD arms wherever they are wanted.
# MODE=base    --ta_lams 0 makes merge:TA@l0 identical to theta_0, so
#              --refine_from ta refines from the PRETRAINED model; all other
#              families are skipped. This is the same mechanism used for the
#              existing from-base results on the twenty-task suite.
#
# Learning-rate grids are centred on each benchmark's n=64 optimum and widened
# by roughly two octaves on each side, because the optimum moves with the
# replay budget (on the shared-output track the APR optimum moved 128 -> 512
# between n=64 and n=16, and grids that were interior at n=64 became
# edge-clipped). Cells whose selected value lands on a grid edge are NOT
# bracketed and must be re-run wider before they enter the paper.
#
# NOTE RegMean is not a --refine_from option in merge_baselines.py, so the
# APR <- RegMean row of Table 1 is not produced here and still needs its own
# run at n=16 (scripts/regmean_baseline.py).

set -euo pipefail
source slurm/common.sh

: "${BENCH:?set BENCH (glue8|clip8|clip20|t5_glue8)}"
: "${MODE:?set MODE (merges|base)}"

# INITS: which merge families to build and refine from. AdaMerging is excluded
# on the shared-output track: its entropy objective assumes a classification
# head, and on a seq2seq LM head the graph detaches, so the backward raises
# "element 0 of tensors does not require grad" (job 25414283). The blank
# AdaMerging cell for that track in Table 1 is therefore a capability gap of
# the baseline, not an experiment left unrun.
case "$BENCH" in
  glue8)     CFG=configs/glue8.yaml;       APR="1 2 4 8 16 32";        NG="0.5 1 2 4 8";      GD="1e-5 5e-5 1e-4 5e-4 1e-3"; INITS="ta ties dareties bc ada"; SKIP="dare della ls" ;;
  clip8)     CFG=configs/clip8.yaml;       APR="2 4 8 16 32";          NG="2 4 8 16";         GD="1e-4 5e-4 1e-3 5e-3";      INITS="ta ties dareties bc ada"; SKIP="dare della ls" ;;
  clip20)    CFG=configs/clip20_l008.yaml; APR="1 2 4 8 16";           NG="1 2 4 8";          GD="1e-4 5e-4 1e-3 5e-3";      INITS="ta ties dareties bc ada"; SKIP="dare della ls" ;;
  t5_glue8)  CFG=configs/t5_glue8.yaml;    APR="64 128 256 512 1024";  NG="4 8 16 32";        GD="1e-3 3e-3 1e-2 3e-2";      INITS="ta ties dareties bc";     SKIP="dare della ls ada" ;;
  *) echo "unknown BENCH=$BENCH" >&2; exit 2 ;;
esac

# The twenty-task suite is scored by absolute accuracy (retention degenerates
# there), so it selects on mean_acc; every other benchmark selects on retention.
SELECT=mean_normret
[ "$BENCH" = clip20 ] && SELECT=mean_acc

OUT="results/compare/grid_n16_${BENCH}_${MODE}.json"
COMMON=(--config "$CFG" --n_probe 16 --steps 5 --select_by "$SELECT"
        --apr_lrs $APR --nogate_lrs $NG --control_gd_lrs $GD --out "$OUT")

echo "[job] BENCH=$BENCH MODE=$MODE n=16 select=$SELECT -> $OUT"

if [ "$MODE" = base ]; then
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --ta_lams 0 \
    --skip_families ties dare dareties della bc ls ada \
    --refine_from ta   # merge:TA@l0 == theta_0
else
  python scripts/merge_baselines.py "${COMMON[@]}" \
    --skip_families $SKIP \
    --refine_from $INITS
fi
