#!/bin/bash
#SBATCH --job-name=drift-probe
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pretrain_drift-%j.out
#SBATCH --error=logs/pretrain_drift-%j.out
#
# Pretraining-capability drift probe on the two NLP benchmarks: does the
# refined model still speak its base model's own objective (masked-LM for
# RoBERTa, prefix-LM for flan-T5) on text the experiments never touch
# (wikitext-2 validation)? Fills the capability-preservation gap the CLIP-20
# held-out study leaves open on the NLP side.
#
# The probed refinement cells are the SPLIT-SELECTED FROM-PRETRAINED WINNERS
# of the n=16+16 deployment tier (Table 2), re-derived deterministically from
# the same buffers (states are not persisted by grid runs). theta_0 and the
# config-lambda TA merge are probed as references.

set -euo pipefail
source slurm/common.sh

python scripts/pretrain_drift.py \
  --config configs/glue8.yaml \
  --n_probe 16 --probe_seed 0 \
  --arms apr:8:50 nogate:8:50 gd:0.001:50 \
  --out results/compare/pretrain_drift_glue8.json

python scripts/pretrain_drift.py \
  --config configs/t5_glue8.yaml \
  --n_probe 16 --probe_seed 0 \
  --arms apr:512:50 nogate:32:5 gd:0.001:50 \
  --out results/compare/pretrain_drift_t5_glue8.json

echo "[done] both drift probes"
