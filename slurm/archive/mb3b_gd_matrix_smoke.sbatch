#!/bin/bash
#SBATCH --job-name=mb3b-gdmx-smoke
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=2:30:00
#SBATCH --output=logs/mb3b_gdmx_smoke-%j.out
#SBATCH --error=logs/mb3b_gdmx_smoke-%j.out

# Smoke test for the decisive mb3b comparison: validates the nogate and
# control-GD arms end-to-end on the causal_lm modality (never exercised there)
# with tiny evals and 2 sweeps, before the ~11h main job spends real budget.

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config /scratch-shared/menglberger/ModelMerging/smoke_mb3b.yaml \
  --ta_lams 0.5 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --steps 2 \
  --apr_lrs 64 \
  --nogate_lrs 64 \
  --control_gd_lrs 1e-4 \
  --out results/compare/mb3b_gd_matrix_smoke.json

echo "[smoke] OK"
