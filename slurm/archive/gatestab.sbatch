#!/bin/bash
#SBATCH --job-name=apr-gatestab
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gatestab-%j.out
#SBATCH --error=logs/gatestab-%j.out
source slurm/common.sh
python scripts/gate_stability.py --config configs/poc3.yaml --steps 5 --lrs 1 8 \
  --out results/compare/gate_stability.json "$@"
