#!/bin/bash
#SBATCH --job-name=mb3b-s14-rnd
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/mb3b_n8_apr_s1235_random-%A_%a.out
#SBATCH --error=logs/mb3b_n8_apr_s1235_random-%A_%a.out
#
# MergeBench-3B n=8+8 follow-up: APR with a smaller sweep-count grid and a
# freshly randomized task order at every sweep. The baseline hyperparameters
# are pinned to the winners from the completed n=8 selection runs, so this job
# isolates APR horizon/order rather than reselecting the merge initialization.
#
# APR uses a constant learning rate throughout: there is no annealing. Ordinary
# GD and ungated APR are also explicitly disabled. Random ordering is seeded by
# the experiment seed and is therefore reproducible.

set -euo pipefail
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    FAMILY=ta
    FAMILY_ARGS=(--skip_families ties dare dareties della bc ls ada
                 --ta_lams 0.5 --refine_from ta)
    ;;
  1)
    FAMILY=ties
    FAMILY_ARGS=(--skip_families ta dare dareties della bc ls ada
                 --ties_densities 0.2 --ties_lams 1.0 --refine_from ties)
    ;;
  2)
    FAMILY=dareties
    FAMILY_ARGS=(--skip_families ta ties dare della bc ls ada
                 --dt_drops 0.5 --dt_trims 0.2 --dt_lams 0.8
                 --refine_from dareties)
    ;;
  3)
    FAMILY=bc
    FAMILY_ARGS=(--skip_families ta ties dare dareties della ls ada
                 --bc_densities 0.2 --bc_outliers 0.01 --bc_lams 0.4
                 --refine_from bc)
    ;;
  4)
    FAMILY=ada
    FAMILY_ARGS=(--skip_families ta ties dare dareties della bc ls
                 --ada_variants task --ada_data probe_buffer
                 --ada_steps 300 --ada_bs 8 --refine_from ada)
    ;;
  *)
    echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT="results/compare/grid_nn_mb3b_${FAMILY}_n8_apr_s1235_random.json"
echo "[job] family=${FAMILY} n=8+8 S={1,2,3,5} random-order APR -> ${OUT}"

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --n_probe 8 --n_select 8 \
  --steps 1 2 3 5 \
  --select_by mean_acc \
  --apr_schedules constant \
  --apr_constant_order random \
  --apr_lrs 16 32 64 128 256 512 1024 2048 \
  --nogate_lrs \
  --gd_lrs \
  --control_gd_lrs \
  --out "${OUT}" \
  "${FAMILY_ARGS[@]}"

echo "[done] -> ${OUT}"
