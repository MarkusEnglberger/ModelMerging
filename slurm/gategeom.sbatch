#!/bin/bash
#SBATCH --job-name=apr-gategeom
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gategeom-%j.out
#SBATCH --error=logs/gategeom-%j.out
source slurm/common.sh
python scripts/gate_geometry.py --config configs/poc3.yaml --out results/compare/gate_geometry.json
