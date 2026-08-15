#!/bin/bash
#SBATCH --job-name=mb3b-n8-smoke
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --output=logs/mb3b_n8_smoke-%j.out
#SBATCH --error=logs/mb3b_n8_smoke-%j.out
#
# Smoke for the mb3b n=8+8 from-pretrained column (slurm/mb3b_n8_base.sbatch).
# The n+n split has been smoke-validated on the encoder path only
# (nn_split_smoke on GLUE-8, which caught selection_obj going unrecorded on
# 4/6 cells); the causal_lm call sites have never run under --n_select. This
# exercises exactly the real job's shape -- --ta_lams 0 from-pretrained init,
# APR + ordinary-GD arms, S grid via constant-schedule checkpoints -- at
# minutes-scale evaluation (configs/mergebench_llama3b_smoke.yaml) and on the
# same H100 memory profile as the real run.
#
# PASS =: job exits 0; report has selection_protocol non-null with
# train_selection_overlap 0; every apr:/gd: cell records selection_obj;
# eval'd cells are only the selected ones (no --eval_all_cells).

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b_smoke.yaml \
  --n_probe 8 --n_select 8 \
  --steps 2 4 \
  --select_by mean_acc \
  --ta_lams 0 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --apr_lrs 128 1024 \
  --control_gd_lrs 1e-4 \
  --out results/compare/mb3b_n8_smoke.json

echo "[done] -> results/compare/mb3b_n8_smoke.json"
