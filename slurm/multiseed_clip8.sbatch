#!/bin/bash
#SBATCH --job-name=apr-ms-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/multiseed_clip8-%j.out
#SBATCH --error=logs/multiseed_clip8-%j.out

# Error bars for the CLIP-8 headline cells over 5 replay-buffer seeds. Only the
# best-config cell of each comparison is re-run (TIES, AdaMerging-layer, GD/ungated/
# random-gate/APR from TA, APR from TIES, APR from AdaMerging-matched). Turns every
# CLIP table entry into mean+/-std so we can tell which orderings survive the ~0.02
# single-seed noise -- notably AdaMerging 0.610 vs APR<-TA 0.588, and APR 0.598 vs
# ungated 0.536. Includes the random-gate matched-density H1 control.
source slurm/common.sh
python scripts/multiseed.py --config configs/clip8.yaml \
  --n_probe 64 --steps 5 --seeds 0 1 2 3 4 \
  --out results/compare/multiseed_clip8.json "$@"
