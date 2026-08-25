#!/bin/bash
#SBATCH --job-name=apgd-pub
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/apgd_published-%j.out
#SBATCH --error=logs/apgd_published-%j.out
#
# DOGE/APGD at its PUBLISHED, pre-registered scale -- a zero-label "published
# recipe" row and a reproduction check of the port. Wei et al. (2025) fix
# eta=0.07 for vision and report 80.7% (8-task ViT-B/32) and 72.5% (20 tasks);
# the split-selected cells in the paper (eta=0.09 / 0.03) score 70.4% / 51.6%.
# Their NLP default eta=0.15 was set on Flan-T5; on RoBERTa the {0.075..0.2}
# grid leaves the merge at theta_0, so GLUE is run at the published 0.15 only
# as a documented reference, not as a fair tuning.
#
# A single eta per run makes "selection" trivial, so the cell is evaluated
# directly; no selection buffer influences the choice.

set -euo pipefail
source slurm/common.sh

run () {  # bench cfg eta
  OUT="results/compare/grid_nn_$1_apgd_published_n16.json"
  echo "[job] DOGE/APGD published recipe BENCH=$1 eta=$3 -> $OUT"
  python scripts/apgd_baseline.py \
    --config "$2" --n_probe 16 --n_select 16 \
    --etas "$3" --out "$OUT"
}

run clip8  configs/clip8.yaml       0.07
run clip20 configs/clip20_l008.yaml 0.07
run glue8  configs/glue8.yaml       0.15

echo "[done] DOGE/APGD published-recipe evaluations"
