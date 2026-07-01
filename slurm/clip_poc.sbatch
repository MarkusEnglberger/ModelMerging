#!/bin/bash
#SBATCH --job-name=apr-clip-poc
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/clip_poc-%j.out
#SBATCH --error=logs/clip_poc-%j.out

# Assets must be pre-fetched on the login node first:
#   python scripts/prefetch_clip.py --tasks eurosat gtsrb mnist
source slurm/common.sh
python scripts/run_merge.py --config configs/clip_poc.yaml "$@"
