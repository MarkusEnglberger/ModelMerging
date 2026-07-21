#!/bin/bash
#SBATCH --job-name=prefetch-clip20
#SBATCH --partition=mcs.default.q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prefetch_clip20-%j.out
#SBATCH --error=logs/prefetch_clip20-%j.out

# Prefetch the 20-task CLIP suite (experts + datasets) into .hf_cache.
#
# NOTE: this deliberately does NOT source slurm/common.sh -- that sets
# HF_HUB_OFFLINE=1, and this job needs the network. Compute nodes DO have
# outbound internet (verified: huggingface.co returns HTTP 200 from a job), so
# prefetching runs here rather than on the login node, whose per-user cgroup is
# capped at 10G and silently OOM-kills dataset conversion. CPU partition: no GPU
# needed. --verify loads every task through load_vision_dataset so a broken
# split/label-key fails here rather than inside the GPU sweep.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
export HF_HOME="$PWD/.hf_cache"
export TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) python=$(which python) HF_HOME=$HF_HOME"
df -h "$HF_HOME" | tail -1

python -u scripts/prefetch_clip.py --suite 20 --verify
rc=$?
echo "[prefetch] exit rc=$rc"
df -h "$HF_HOME" | tail -1
exit $rc
