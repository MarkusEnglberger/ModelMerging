#!/bin/bash
#SBATCH --job-name=bench-gencache
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --output=logs/bench_gen_cache-%j.out
#SBATCH --error=logs/bench_gen_cache-%j.out
#
# Micro-benchmark: does the MergeBench experts' use_cache=False (baked into
# their config.json / generation_config.json) make HF generate() run without
# a KV cache in our eval path? Times one batch with and without cache.

set -euo pipefail
source slurm/common.sh
nvidia-smi --query-gpu=name,power.limit --format=csv
python scripts/bench_gen_cache.py --batch 48 --max_new 256 &
PID=$!
# sample utilization / power every 15 s while the benchmark runs
while kill -0 $PID 2>/dev/null; do
  nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
  sleep 15
done
wait $PID
