#!/bin/bash
#SBATCH --job-name=apgd-paperk
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/apgd_paper_subspace-%j.out
#SBATCH --error=logs/apgd_paper_subspace-%j.out
#
# REPRODUCTION CHECK: DOGE/APGD at the published recipe, with the subspace
# basis size taken from the PAPER rather than from the released config.
#
# Wei et al. (2025), implementation details: "The subspace basis size k is
# simply defined as the rank of each task vector divided by the number of
# tasks (i.e., 8)."  Their released config/method/DOGE_TA/DOGE_TA.yaml instead
# hardcodes subspace: 6, and the code computes rank/subspace. Our port copied
# the config (6). At 8 tasks that is 6 vs 8; at 20 tasks it is 6 vs 20, i.e. a
# 3.4x oversized projection subspace -- and the 20-task suite is where our
# reproduction is furthest off (41.1% against a published 72.5%).
#
# Everything else is the published recipe: eta=0.07 vision / 0.15 NLP, 400
# iterations, lr 1e-4, keep_density 0.30. A single eta makes "selection"
# trivial, so the cell is evaluated directly; no selection buffer is involved.
#
# Reference numbers to beat: 80.7% (CLIP-8), 72.5% (CLIP-20).

set -euo pipefail
source slurm/common.sh

run () {  # bench cfg eta divisor
  OUT="results/compare/grid_nn_$1_apgd_paperk_n16.json"
  echo "[job] DOGE published recipe, paper subspace rule: BENCH=$1 eta=$3 divisor=$4 -> $OUT"
  python scripts/apgd_baseline.py \
    --config "$2" --n_probe 16 --n_select 16 \
    --etas "$3" --subspace_divisor "$4" --out "$OUT"
}

run clip8  configs/clip8.yaml       0.07 8
run clip20 configs/clip20_l008.yaml 0.07 20
run glue8  configs/glue8.yaml       0.15 8

echo "[done] DOGE/APGD paper-subspace evaluations"
