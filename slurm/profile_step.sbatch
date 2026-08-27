#!/bin/bash
#SBATCH --job-name=profile-step
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=logs/profile_step-%j.out
#SBATCH --error=logs/profile_step-%j.out
#
# Per-component timing of one refine() task-step and of an evaluation pass,
# for the two eight-task suites (scripts/profile_step.py). Same node shape as
# the protocol runs so the numbers transfer.

set -euo pipefail
source slurm/common.sh
source slurm/args.sh "$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CONFIGS:=clip8 glue8}"

for c in ${CONFIGS}; do
  echo; echo "################ ${c} ################"
  python scripts/profile_step.py --config "configs/${c}.yaml" \
    --out "results/profile/${c}_step.json"
done
echo "[done]"
