#!/bin/bash
# MaTS/T0 shared-output track: bracket the refinement grid and replicate over
# replay-buffer seeds. Closes the two gaps the paper lists for Section~\ref{sec:t5}
# ("single-seed with refinement grids whose optima are not yet bracketed").
#
# Every cell of the earlier disjoint-selection runs selected S=20, the MINIMUM of
# the {20,40,80} horizon grid, and several selected the extreme learning rate
# (128 = max for Fisher/Localize&Stitch, 32 = min for the TIES families). The
# grid below extends the horizon down to 3 and opens the learning rate on both
# sides, so the optimum can sit in the interior.
#
# Cost note: APR re-runs a trajectory per (steps, lr) cell while GD reuses
# checkpoints along one constant-LR run, so APR cost scales with
# (sum of steps) x (number of lrs). Shortening the horizons pays for the wider
# learning-rate axis: 38x6 = 228 APR step-units here against 140x3 = 420 before.
#
# Usage:
#   sbatch --export=ALL,SEED=0,GROUP=a slurm/mats_bracket_multiseed.sbatch
#
#SBATCH --job-name=mats-brkt
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/mats_bracket-%j.out
#SBATCH --error=logs/mats_bracket-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/common.sh
export HF_DATASETS_OFFLINE=1
export MATS_DATA_DIR="$PWD/.mats_data"

: "${SEED:?set SEED (replay-buffer draw)}"
: "${GROUP:?set GROUP (a|b|c)}"

# The selection buffer is drawn disjointly from the probe buffer; give it its
# own stream so the two are independent as well as non-overlapping.
SEL_SEED=$((SEED + 1))

case "$GROUP" in
  a) FAMILIES=(Average TaskArithmetic TIES) ;;
  b) FAMILIES=(DARE-TA DARE-TIES) ;;
  c) FAMILIES=(Fisher-64 'Localize&Stitch') ;;
  *) echo "unknown GROUP=$GROUP (expected a, b or c)" >&2; exit 2 ;;
esac

OUT="results/mats_t5_8/bracket_s${SEED}${GROUP}.json"
echo "[job] seed=${SEED} selection_seed=${SEL_SEED} group=${GROUP}"
echo "[job] families=${FAMILIES[*]}"
echo "[job] out=${OUT}"

python scripts/mats_fair_compare.py \
  --config configs/mats_t5_8.yaml \
  --n_probe 64 \
  --n_select 64 \
  --probe_seed "${SEED}" \
  --selection_seed "${SEL_SEED}" \
  --grad_batch_size 16 \
  --apr_lrs 16 32 64 128 256 512 \
  --apr_steps 3 5 10 20 \
  --gd_lrs 0.5 1 2 3 6 \
  --gd_steps 3 5 10 20 \
  --families "${FAMILIES[@]}" \
  --out "${OUT}"
