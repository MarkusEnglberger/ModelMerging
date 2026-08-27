#!/bin/bash
#SBATCH --job-name=pin-glue8
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pin_cell_glue8-%j.out
#SBATCH --error=logs/pin_cell_glue8-%j.out
#
# Does a low-scoring GLUE-8 draw fail because SELECTION could not see the
# signal, or because the BUFFER has none? Pin cells instead of selecting them.
#
# The full 2x2 at each budget: the good draw's cell applied to the two bad
# draws (does a good cell transfer?), and a bad draw's cell applied to the good
# draw (is the bad cell bad everywhere?). The diagonal reproduces the reported
# runs and is the control -- it must match them, since a pinned refit at the
# selected cell IS the reported refit.
#
#   B=16: good = draw0 (eta=4, S=50, NormRet +0.305)
#         bad  = draw1 (eta=16, S=5, +0.023), draw2 (eta=0.5, S=100, +0.060)
#   B=32: good = draw0 (eta=2, S=100, +0.369)
#         bad  = draw1 (eta=4, S=20, +0.051)

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[job] pinned-cell probe on GLUE-8"

# ---- B=16: draw0's cell onto draws 1,2 ; each bad cell onto draw0 ----
python scripts/pin_cell_probe.py \
  --config configs/glue8.yaml --budget 16 \
  --cells 101:4:50 102:4:50 100:16:5 100:0.5:100 100:4:50 \
  --out results/compare/pin_glue8_B16.json

# ---- B=32: draw0's cell onto draw1 ; draw1's cell onto draws 0,2 ----
python scripts/pin_cell_probe.py \
  --config configs/glue8.yaml --budget 32 \
  --cells 101:2:100 100:4:20 102:4:20 100:2:100 \
  --out results/compare/pin_glue8_B32.json

echo "[done]"
