#!/bin/bash
# Full unattended clip20 pipeline: prefetch (network, login node) -> submit the
# two single-seed jobs. Run inside tmux so it survives session teardown.
set -uo pipefail
cd /vast.mnt/home/20250638/ModelMerging
source .venv/bin/activate
export HF_HOME="$PWD/.hf_cache"
LOG=logs/clip20_driver.log
echo "[$(date)] pipeline start" >> "$LOG"

python -u scripts/prefetch_clip.py --suite 20 --verify >> logs/prefetch_clip20.log 2>&1
rc=$?
echo "[$(date)] prefetch exited rc=$rc" >> "$LOG"
if [ $rc -ne 0 ] || ! grep -q "^\[prefetch\] done\.$" logs/prefetch_clip20.log; then
  echo "[$(date)] ABORT: prefetch failed; not submitting" >> "$LOG"
  grep -E "FAILED" logs/prefetch_clip20.log | tail -20 >> "$LOG"
  exit 1
fi
echo "[$(date)] prefetch OK ($(grep -c ': OK ' logs/prefetch_clip20.log) verified); submitting" >> "$LOG"
sbatch slurm/merge_baselines_clip20.sbatch  >> "$LOG" 2>&1
sbatch slurm/calibrate_refine_clip20.sbatch >> "$LOG" 2>&1
squeue -u "$USER" >> "$LOG" 2>&1
echo "[$(date)] pipeline done" >> "$LOG"
