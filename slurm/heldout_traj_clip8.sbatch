#!/bin/bash
#SBATCH --job-name=heldout-traj-clip8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/heldout_traj_clip8-%A_%a.out
#SBATCH --error=logs/heldout_traj_clip8-%A_%a.out
#
# Held-out retention along the XGD refinement trajectory: ONE DRAW PER ARRAY
# TASK. Re-runs the protocol-v3 refit of the draw's selected cell from the
# pretrained model with per-sweep checkpoints, then evaluates every
# checkpoint on the 8 train tasks and zero-shot on the 7 reportable held-out
# CLIP-20 tasks (the tab:heldout8 split). LRS lists the selected learning
# rate per draw (array index d uses the d-th entry; buffer seed = SEED0+d).
#
#   sbatch --array=0-2 slurm/heldout_traj_clip8.sbatch \
#          BUDGET=32 "LRS=32 32 32" S=50

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${BUDGET:=32}"
: "${LRS:=32 32 32}"           # selected lr per draw, indexed by array id
: "${S:=50}"                   # the selected horizon (last checkpoint)
: "${CHECKPOINTS:=1 2 5 10 20 30 40 ${S}}"
: "${SUFFIX:=_v3fresh}"        # winners dir suffix (reproduction check)
: "${SEED0:=103}"              # buffer seed of array task 0 (fresh draws)

DRAW=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((SEED0 + DRAW))
read -ra LR_ARR <<< "${LRS}"
LR=${LR_ARR[$DRAW]}
OUT="results/compare/heldout_traj_clip8_B${BUDGET}_draw${DRAW}.json"

WINNER="results/cv_winners/clip8_B${BUDGET}_pretrained${SUFFIX}/draw${DRAW}/apr_pretrained_draw0.pt"
WINFLAG=()
if [ -f "${WINNER}" ]; then WINFLAG=(--winner "${WINNER}"); fi

echo "[job] XGD trajectory probe: B=${BUDGET} draw=${DRAW} (seed ${SEED}) lr=${LR}" \
     "checkpoints=[${CHECKPOINTS}] -> ${OUT}"
guard_output "${OUT}"

# shellcheck disable=SC2086  # CHECKPOINTS is a word list on purpose
python scripts/heldout_trajectory.py \
  --config configs/clip20.yaml \
  --budget "${BUDGET}" --buffer_seed "${SEED}" --lr "${LR}" \
  --checkpoints ${CHECKPOINTS} \
  --order random \
  "${WINFLAG[@]}" \
  --out "${OUT}"

echo "[done] -> ${OUT}"
