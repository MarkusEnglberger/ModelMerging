#!/bin/bash
#SBATCH --job-name=apr-msl-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ms_longapr_glue8-%j.out
#SBATCH --error=logs/ms_longapr_glue8-%j.out

# Multi-seed (5 replay seeds) replication of the long-horizon-vs-calibrate comparison
# on GLUE-8, at n=32 (where the single-seed control FLIPPED: annealed 0.620 beat
# two-stage 0.563) and n=64 (where two-stage led 0.662 vs 0.619, also within the ~0.06
# seed band). Arms: apr-short / apr-anneal(S=30,random,cosine) / calibrated / apr<-calib.
source slurm/common.sh
python scripts/multiseed_longapr.py --config configs/glue8.yaml \
  --n_probe 32 --seeds 0 1 2 3 4 --short_steps 5 --long_steps 30 --lr 16 \
  --cal_lr 0.01 --cal_l2 1e-3 --cal_steps 300 \
  --out results/compare/ms_longapr_glue8_n32.json
python scripts/multiseed_longapr.py --config configs/glue8.yaml \
  --n_probe 64 --seeds 0 1 2 3 4 --short_steps 5 --long_steps 30 --lr 16 \
  --cal_lr 0.01 --cal_l2 1e-3 --cal_steps 300 \
  --out results/compare/ms_longapr_glue8_n64.json
