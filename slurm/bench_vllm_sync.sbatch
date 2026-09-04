#!/bin/bash
#SBATCH --job-name=bench-vllm
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:50:00
#SBATCH --output=logs/bench_vllm_sync-%j.out
#SBATCH --error=logs/bench_vllm_sync-%j.out
#
# vLLM prototype benchmark (scripts/bench_vllm_sync.py) in the SEPARATE
# scratch venv (torch 2.4.0 / vllm 0.6.3). Production .venv is not used.

set -euo pipefail
source /scratch-shared/menglberger/ModelMerging/venv_vllm/bin/activate
export HF_HOME="$PWD/.hf_cache"
export NLTK_DATA=/scratch-shared/menglberger/ModelMerging/nltk_data
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
echo "host=$(hostname) python=$(which python)"
python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/bench_vllm_sync.py "$@"
