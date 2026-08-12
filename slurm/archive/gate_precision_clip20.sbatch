#!/bin/bash
#SBATCH --job-name=gateprec-c20
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/gateprec_clip20-%j.out
#SBATCH --error=logs/gateprec_clip20-%j.out

# GATE-PRECISION DIAGNOSTIC on the 20-task suite at the tuned TA merge
# (lam=0.08): does thresholding help, and by WHICH statistic? Buckets the
# g*v<0 coordinates by percentile of |g*v| (the topk gate's quantity), |g|
# (significance on the only noisy factor), and per-tensor-normalized |g|,
# then measures the actual replay-loss change of a small toward-expert step
# restricted to each bucket vs the first-order prediction. Identical
# intervention across statistics -> directly comparable.
# Decides whether the topk_g gate sweep (gate_topkg_clip20.sbatch) launches:
# if precision is flat across |g| buckets, no threshold can help and the
# ungated-anchor story stands.
source slurm/common.sh
python scripts/gate_precision.py --config configs/clip20_l008.yaml \
  --n_probe 64 --alphas 0.05 0.2 --stats gv g gnorm \
  --out results/compare/gate_precision_clip20.json
