#!/bin/bash
#SBATCH --job-name=cv-glue8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cv_protocol_glue8-%j.out
#SBATCH --error=logs/cv_protocol_glue8-%j.out
#
# Protocol v2 on multi-head GLUE-8 (RoBERTa-base): equal total label budget,
# 2-way split select-then-refit, D independent buffer draws, grids auto-
# extended to interiority, refit winners persisted for the retention probes.
# See scripts/cv_protocol.py. Learning-rate grids follow the earlier GLUE-8
# campaign (paper Appendix); ablations run from the pretrained model only.

set -euo pipefail
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${BUDGET:=32}"
: "${FOLDS:=2}"
: "${DRAWS:=3}"
: "${INIT:=pretrained}"

OUT="results/compare/cv_glue8_B${BUDGET}_${INIT}.json"
WIN="results/cv_winners/glue8_B${BUDGET}_${INIT}"

echo "[job] GLUE-8 protocol v2: B=${BUDGET} folds=${FOLDS} draws=${DRAWS} init=${INIT} -> ${OUT}"

python scripts/cv_protocol.py \
  --config configs/glue8.yaml \
  --budget "${BUDGET}" --folds "${FOLDS}" --draws "${DRAWS}" \
  --init "${INIT}" \
  --arms apr nogate gd \
  --apr_lrs 0.5 1 2 4 8 16 32 \
  --nogate_lrs 0.25 0.5 1 2 4 8 \
  --gd_lrs 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 \
  --steps 5 20 50 100 \
  --order random \
  --max_extensions 2 \
  --save_winners "${WIN}" \
  --out "${OUT}"

echo "[done] -> ${OUT}"
