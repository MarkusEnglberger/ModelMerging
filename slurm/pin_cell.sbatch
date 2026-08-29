#!/bin/bash
#SBATCH --job-name=pin-cell
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pin-%x-%j.out
#SBATCH --error=logs/pin-%x-%j.out
#
# Refit and evaluate GIVEN (buffer seed, eta, S) cells from the pretrained
# model, exactly as cv_protocol's refit does. Used to evaluate cells that a
# selection rule chose on stored grids without re-running the sweeps, and for
# the selection-failure diagnostics. Parameters are positional KEY=VALUE
# arguments (slurm/args.sh):
#
#   sbatch -J pin-glue8 slurm/pin_cell.sbatch CONFIG=configs/glue8.yaml \
#          BUDGET=16 "CELLS=101:16:50 102:4:50" OUT=results/compare/pin_x.json

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CONFIG:?CONFIG=configs/<suite>.yaml is required}"
: "${BUDGET:?BUDGET=<labels per task> is required}"
: "${CELLS:?CELLS=\"SEED:ETA:S ...\" is required}"
: "${OUT:?OUT=<results json> is required}"
: "${ARM:=apr}"

echo "[job] pinned cells: ${CONFIG} B=${BUDGET} arm=${ARM} cells=[${CELLS}] -> ${OUT}"
guard_output "${OUT}"

# shellcheck disable=SC2086  # CELLS is a word list on purpose
python scripts/pin_cell_probe.py \
  --config "${CONFIG}" --budget "${BUDGET}" --arm "${ARM}" \
  --cells ${CELLS} \
  --out "${OUT}"

echo "[done] -> ${OUT}"
