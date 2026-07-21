#!/bin/bash
# Wait for the running clip20 prefetch to finish, then submit the sweep -- but
# only if every task prefetched+verified cleanly. Logs to logs/clip20_driver.log.
set -uo pipefail
cd /vast.mnt/home/20250638/ModelMerging
LOG=logs/clip20_driver.log
PREF_LOG=logs/prefetch_clip20.log
PREF_PID="${1:?usage: _run_clip20_after_prefetch.sh <prefetch_pid>}"

echo "[$(date)] watching prefetch pid $PREF_PID" >> "$LOG"
while kill -0 "$PREF_PID" 2>/dev/null; do sleep 30; done
echo "[$(date)] prefetch process exited" >> "$LOG"

if ! grep -q "^\[prefetch\] done\.$" "$PREF_LOG"; then
  echo "[$(date)] ABORT: prefetch did not report 'done.' -- check $PREF_LOG" >> "$LOG"
  grep -E "FAILED|failure" "$PREF_LOG" | tail -20 >> "$LOG"
  exit 1
fi
n_ok=$(grep -c ": OK " "$PREF_LOG")
echo "[$(date)] prefetch OK ($n_ok tasks verified); submitting single-seed jobs" >> "$LOG"
# Principled inits (tuned/learned lambda), not the fixed-lambda=0.3 TA compare:
#   merge_baselines -> TA(lambda swept low) / TIES / AdaMerging, APR refined from each
#   calibrate_refine -> per-tensor calibrated init + APR (CLIP headline)
sbatch slurm/merge_baselines_clip20.sbatch >> "$LOG" 2>&1
sbatch slurm/calibrate_refine_clip20.sbatch >> "$LOG" 2>&1
echo "[$(date)] submitted; squeue:" >> "$LOG"
squeue -u "$USER" >> "$LOG" 2>&1
