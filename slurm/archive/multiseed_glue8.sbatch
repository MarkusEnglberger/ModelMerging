#!/bin/bash
#SBATCH --job-name=apr-ms-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/multiseed_glue8-%j.out
#SBATCH --error=logs/multiseed_glue8-%j.out

# Error bars for the GLUE-8 headline cells over 5 replay-buffer seeds (TIES, GD/ungated/
# random-gate/APR from TA, APR from DARE-TIES). Tests whether the gate-neutral GLUE story
# (ungated 0.637 vs APR 0.595) and APR<-DARETIES 0.610 survive the single-seed noise.
# Includes the random-gate matched-density H1 control.
source slurm/common.sh
python scripts/multiseed.py --config configs/glue8.yaml \
  --n_probe 64 --steps 5 --seeds 0 1 2 3 4 \
  --out results/compare/multiseed_glue8.json "$@"
