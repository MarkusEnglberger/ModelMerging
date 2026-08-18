#!/bin/bash
#SBATCH --job-name=mb3b-n8-mapr
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/mb3b_n8_merges_apr-%A_%a.out
#SBATCH --error=logs/mb3b_n8_merges_apr-%A_%a.out
#
# Missing MergeBench-3B n=8+8 composition cells: each array task runs one
# canonical merge family (standalone baseline) and APR on top. The pretrained
# initialization is already complete in grid_nn_mb3b_base_n8_s100.json.
#
# Intentionally absent: ordinary GD and the ungated anchored ablation. Passing
# empty lists explicitly below prevents either control from being enabled by a
# future default change.
#
# The five families match Table main8:
#   0 task arithmetic, 1 TIES, 2 DARE-TIES, 3 Breadcrumbs,
#   4 matched-budget AdaMerging.
#
# All merge hyperparameters and APR (lr, S) cells are selected on the disjoint
# 8-example selection buffer. Only the selected standalone baseline and selected
# APR cell are evaluated on the benchmark split. The APR rate grid brackets the
# previous decoder composition sweep (which improved through lr=64 at S=5) on
# both sides while sharing exactly the same grid across initializations.

set -euo pipefail
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    FAMILY=ta
    FAMILY_ARGS=(--skip_families ties dare dareties della bc ls ada
                 --refine_from ta)
    ;;
  1)
    FAMILY=ties
    FAMILY_ARGS=(--skip_families ta dare dareties della bc ls ada
                 --refine_from ties)
    ;;
  2)
    FAMILY=dareties
    FAMILY_ARGS=(--skip_families ta ties dare della bc ls ada
                 --refine_from dareties)
    ;;
  3)
    FAMILY=bc
    FAMILY_ARGS=(--skip_families ta ties dare dareties della ls ada
                 --refine_from bc)
    ;;
  4)
    FAMILY=ada
    FAMILY_ARGS=(--skip_families ta ties dare dareties della bc ls
                 --refine_from ada)
    ;;
  *)
    echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT="results/compare/grid_nn_mb3b_${FAMILY}_n8_apr.json"
echo "[job] family=${FAMILY} n=8+8 APR-only composition -> ${OUT}"

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --n_probe 8 --n_select 8 \
  --steps 5 20 50 100 \
  --select_by mean_acc \
  --ta_lams 0.2 0.3 0.4 0.5 \
  --ties_densities 0.1 0.2 --ties_lams 0.8 1.0 \
  --dt_drops 0.5 --dt_trims 0.1 0.2 --dt_lams 0.8 1.0 \
  --bc_densities 0.1 0.2 --bc_outliers 0.01 0.05 --bc_lams 0.4 \
  --ada_variants task layer --ada_data probe_buffer \
  --ada_steps 300 --ada_bs 8 \
  --apr_lrs 1 2 4 8 16 32 64 128 256 \
  --nogate_lrs \
  --gd_lrs \
  --control_gd_lrs \
  --out "${OUT}" \
  "${FAMILY_ARGS[@]}"

echo "[done] -> ${OUT}"
