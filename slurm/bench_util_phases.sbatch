#!/bin/bash
#SBATCH --job-name=bench-util
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_util_phases-%j.out
#SBATCH --error=logs/bench_util_phases-%j.out
#
# Sample GPU utilization + power across a whole MergeBench run (loading,
# refinement/gradient sweeps, generation eval) so the utilization claim covers
# the run rather than one phase. Samples land in a CSV keyed by timestamp; the
# main log's phase markers let each sample be attributed to a phase.

set -euo pipefail
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CSV="logs/bench_util_phases-${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used \
           --format=csv,noheader -l 2 > "$CSV" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

date +"[phase] start %H:%M:%S"
python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --n_probe 8 --n_select 8 \
  --steps 5 \
  --select_by mean_acc \
  --ta_lams 0.3 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --apr_lrs 1024 \
  --out results/compare/mb3b_util_probe.json
date +"[phase] end %H:%M:%S"

kill $SMI 2>/dev/null || true
echo "=== utilization summary (per phase) ==="
python scripts/util_phases_report.py "logs/bench_util_phases-${SLURM_JOB_ID}.out"
