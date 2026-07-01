#!/bin/bash
# Shared SLURM environment setup, sourced by job scripts.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate
export HF_HOME="$PWD/.hf_cache"
# assets are pre-fetched on the login node; run offline on compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) python=$(which python)"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
