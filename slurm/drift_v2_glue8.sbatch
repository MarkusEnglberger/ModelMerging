#!/bin/bash
#SBATCH --job-name=drift-v2-glue8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/drift_v2_glue8-%j.out
#SBATCH --error=logs/drift_v2_glue8-%j.out
#
# Pretraining-objective drift for the protocol-v2 GLUE-8 models (Table
# tab:drift). Every refined state is loaded from cv_protocol.py --save_winners
# output, so the probed model IS the one Table tab:cv-v2 reports: nothing is
# re-derived, and all three buffer draws are probed, which turns the current
# single-seed table into one with error bars.
#
# MLM loss is measured on WikiText-2 validation under RoBERTa's own pretrained
# MLM head, which no experiment modifies, and that text is never used for
# refinement or selection -- so any rise over theta_0 is encoder drift alone.
#
# The merge rows use the hyperparameters v2 selected on the B=32 budget
# (TA l0.3, TIES d0.2/l1, DARE-TIES dd0.5/t0.1/l1, BC d0.2/o0.01/l0.4);
# merges are deterministic given the checkpoints, hence one row each.
#
# Protocol v3 (fresh seeds 103-105, metric selection) reuses this job with
# positional overrides: SUFFIX names the winners directory, LABEL the output,
# MERGES lists every merge cell v3 selected on any draw, e.g.
#   sbatch slurm/drift_v2_glue8.sbatch SUFFIX=_v3fresh LABEL=v3 \
#          "MERGES=ties:0.1:1.0 dareties:0.5:0.1:1.0 bc:0.2:0.01:0.4 bc:0.2:0.05:0.4"

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${BUDGET:=32}"
: "${SUFFIX:=}"                # winners dir suffix (_v3fresh for protocol v3)
: "${LABEL:=v2}"               # output file label
: "${MERGES:=ties:0.2:1.0 dareties:0.5:0.1:1.0 bc:0.2:0.01:0.4}"
W="results/cv_winners/glue8_B${BUDGET}_pretrained${SUFFIX}"
OUT="results/compare/drift_${LABEL}_glue8_B${BUDGET}.json"

STATES=()
for d in 0 1 2; do
  for arm in apr nogate gd; do
    # cv_protocol tags the file by its own draw index, which is 0 in every
    # per-draw job; the enclosing directory carries the real draw number.
    f="${W}/draw${d}/${arm}_pretrained_draw0.pt"
    [ -f "$f" ] && STATES+=("${arm}_draw${d}=${f}")
  done
done
echo "[job] probing ${#STATES[@]} ${LABEL} winner states + merges [${MERGES}] -> ${OUT}"
printf '       %s\n' "${STATES[@]}"
guard_output "${OUT}"

# shellcheck disable=SC2086  # MERGES is a word list on purpose
python scripts/pretrain_drift.py \
  --config configs/glue8.yaml \
  --n_probe "${BUDGET}" \
  --state_files "${STATES[@]}" \
  --merges ${MERGES} \
  --out "${OUT}"

echo "[done] -> ${OUT}"
