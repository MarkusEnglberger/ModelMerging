#!/bin/bash
#SBATCH --job-name=ada-trans-nn
#SBATCH --partition=mcs.gpu.q
#SBATCH --constraint=v100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ada_transductive_nn-%A_%a.out
#SBATCH --error=logs/ada_transductive_nn-%A_%a.out
#
# Standard transductive AdaMerging and APR composed on top, for the 16+16 and
# 8+8 split-selection protocols. AdaMerging adapts on eval_ds, as in its
# canonical formulation; APR still trains on n labeled replay examples and is
# selected using the disjoint n-example selection buffer. The evaluation split
# is necessarily no longer untouched because the initialization is transductive.
#
# One array task handles both budgets for a benchmark sequentially. This avoids
# concurrent writes to the AdaMerging cache, whose transductive state is shared
# across replay budgets.

set -euo pipefail
source slurm/common.sh

case "${SLURM_ARRAY_TASK_ID}" in
  0) BENCH=glue8;  CFG=configs/glue8.yaml;       SELECT=mean_normret ;;
  1) BENCH=clip8;  CFG=configs/clip8.yaml;       SELECT=mean_normret ;;
  2) BENCH=clip20; CFG=configs/clip20_l008.yaml; SELECT=mean_acc ;;
  *) echo "unknown array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

run_budget() {
  local n="$1"
  local apr steps out

  # Match the grids used by the result files currently selected by
  # make_table_n16.py. The n=8 GLUE-8/CLIP-8 edge files extended S to 100 and
  # widened the learning-rate bracket; CLIP-20 retained the canonical grid.
  if [ "$n" = 16 ]; then
    steps="5 20 50"
    case "$BENCH" in
      glue8)  apr="1 2 4 8 16 32" ;;
      clip8)  apr="2 4 8 16 32" ;;
      clip20) apr="1 2 4 8 16" ;;
    esac
  else
    case "$BENCH" in
      glue8|clip8) apr="0.5 1 2 4 8 16 32"; steps="5 20 50 100" ;;
      clip20)      apr="1 2 4 8 16";        steps="5 20 50" ;;
    esac
  fi

  out="results/compare/grid_nn_${BENCH}_ada_transductive_n${n}.json"
  echo "[job] BENCH=$BENCH n=${n}+${n} transductive AdaMerging; S in {$steps} -> $out"

  python scripts/merge_baselines.py \
    --config "$CFG" \
    --n_probe "$n" \
    --n_select "$n" \
    --steps $steps \
    --select_by "$SELECT" \
    --apr_lrs $apr \
    --ada_data eval_ds \
    --skip_families ta ties dare dareties della bc ls \
    --refine_from ada \
    --out "$out"
}

# Populate/reuse the benchmark-level transductive AdaMerging cache first, then
# rerun only APR's replay/selection procedure at the smaller budget.
run_budget 16
run_budget 8
