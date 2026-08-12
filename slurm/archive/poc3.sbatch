#!/bin/bash
#SBATCH --job-name=apr-poc3
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/poc3-%j.out
#SBATCH --error=logs/poc3-%j.out

source slurm/common.sh
python scripts/run_merge.py --config configs/poc3.yaml "$@"
