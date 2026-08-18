#!/bin/bash
#SBATCH --job-name=apr-t5-glue8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/t5_glue8-%j.out
#SBATCH --error=logs/t5_glue8-%j.out

# Shared-head T5 track: fair 3-family sweep (APR / ungated / ordinary GD) on flan-T5
# text-to-text GLUE-8, same protocol as glue8_compare. flan-t5-base = 248M params, so
# this fits the V100s -- no need to wait for the H100.
# lr grids start from the RoBERTa GLUE-8 optima but shifted down one notch: the merged
# space now includes decoder+embeddings, and vdist keeps high-lr cells bounded.
# [eval0] check: each expert must beat base clearly on its own task; if not, the
# templates in src/apr/t5_gen.py have drifted from the experts' training templates.
# Requires: python scripts/prefetch_t5.py
source slurm/common.sh
python scripts/compare_baselines.py --config configs/t5_glue8.yaml \
  --steps 5 --clip_mode vdist \
  --apr_lrs 4 8 16 32 \
  --nogate_lrs 1 2 4 8 \
  --gd_lrs 5e-5 1e-4 5e-4 1e-3 \
  --out results/compare/t5_glue8_compare.json "$@"
