#!/bin/bash
#SBATCH --job-name=apr-gsig-clip8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gate_signal_clip8-%j.out
#SBATCH --error=logs/gate_signal_clip8-%j.out

# Gate-signal investigation on CLIP-8 (user push-back: random-mask parity is
# counterintuitive; is attribution signal concentrated in heavy |g*v| coords?).
# 1) gate_precision: measured vs predicted d-loss per |g*v|-percentile bucket.
# 2) topk gate vs sign gate vs density-matched random, 3 seeds, S=5, lr16.
source slurm/common.sh
python scripts/gate_precision.py --config configs/clip8.yaml \
  --n_probe 64 --alphas 0.05 0.2 \
  --out results/compare/gate_precision_clip8.json
python scripts/topk_gate.py --config configs/clip8.yaml \
  --n_probe 64 --seeds 0 1 2 --steps 5 --lr 8 --topk_fracs 0.01 0.05 0.1 \
  --out results/compare/topk_gate_clip8.json
