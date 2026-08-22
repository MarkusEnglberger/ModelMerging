#!/bin/bash
#SBATCH --job-name=ms-glue8-n16
# AdaMerging's backward pass exceeds the 6 GB A30 MIG slice. Use the same
# full-GPU queue as the earlier successful GLUE-8/AdaMerging experiments.
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --time=20:00:00
#SBATCH --array=0-4%3
#SBATCH --output=logs/ms_glue8_n16-%A_%a.out
#SBATCH --error=logs/ms_glue8_n16-%A_%a.out

# Pinned verification of every replay-dependent cell in the headline GLUE-8
# n=16 column.  Hyperparameters and refinement RNG remain at their seed-0
# selections; only the replay-buffer draw varies across the array.
set -euo pipefail
source slurm/common.sh

SEED=${SLURM_ARRAY_TASK_ID}
OUT="results/compare/ms_glue8_n16_seed${SEED}.json"
echo "[job] pinned GLUE-8 n=16 column, replay seed=${SEED} -> ${OUT}"
python scripts/multiseed_verify_glue8_n16.py \
  --probe_seed "${SEED}" \
  --out "${OUT}"
