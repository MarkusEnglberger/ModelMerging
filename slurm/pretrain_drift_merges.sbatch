#!/bin/bash
#SBATCH --job-name=drift-merges
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pretrain_drift_merges-%j.out
#SBATCH --error=logs/pretrain_drift_merges-%j.out
#
# Extends the GLUE-8 pretraining-drift probe (paper Table 3) beyond task
# arithmetic: probes the SPLIT-SELECTED merge baselines of the 16+16 grid
# (TIES@d0.2,l1; DARE-TIES@dd0.5,t0.1,l1; Breadcrumbs@d0.2,o0.01,l0.4;
# AdaMerging-task, matched budget -- see best_per_family in
# results/compare/grid_nn_glue8_merges_n16.json) with the same masked-LM loss
# on wikitext-2 validation, so the retention comparison is against the merge
# methods rather than only the task-arithmetic reference. The theta_0 / TA
# references and the three from-pretrained arms are re-derived alongside so
# the output file is self-consistent.

set -euo pipefail
source slurm/common.sh

python scripts/pretrain_drift.py \
  --config configs/glue8.yaml \
  --n_probe 16 --probe_seed 0 \
  --arms apr:8:50 nogate:8:50 gd:0.001:50 \
  --merges ties:0.2:1 dareties:0.5:0.1:1 bc:0.2:0.01:0.4 ada:task \
  --out results/compare/pretrain_drift_glue8_merges.json

echo "[done] drift probe with merge baselines"
