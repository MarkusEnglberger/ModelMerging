#!/bin/bash
#SBATCH --job-name=t5-s200-pair
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=logs/t5_matched_s200_paired-%j.out
#SBATCH --error=logs/t5_matched_s200_paired-%j.out
#
# Both t2t merges pairs CO-SCHEDULED on one GPU. The single-process runs leave
# the GPU ~40% idle (58% util measured on the CLIP-8 analogue: 16-example
# batches and per-coordinate update ops cannot saturate an A100, and eval /
# snapshot phases are CPU-bound), so two runs time-slice into one GPU billing
# share instead of two. Peak RSS of a t5 merges run is ~36G (job 25447776), so
# two fit in the 100G single-share request with headroom.
#
# Expected wall: each pair alone is ~22-24h; at ~58% GPU demand apiece the
# contention factor is ~1.2, so ~28h combined -- billed at ONE share instead
# of two shares x 22h. Worst case (both fully GPU-bound) is ~44h, inside the
# 48h limit, and even then the billing is no worse than the two separate jobs.
#
# Each process gets its own log and its own output JSON; the job fails if
# either process fails, but one failing does not kill the other mid-write.
#
# Grids and protocol are inherited verbatim from slurm/t5_matched_s200.sbatch
# (which remains the single-run form for base mode and reruns).

set -uo pipefail
source slurm/common.sh

run_pair () {
  local inits="$1"
  local tag; tag=$(echo "$inits" | tr -d ' ')
  MODE=merges INITS="$inits" bash slurm/t5_matched_s200_inner.sh \
    > "logs/t5_s200_${tag}_${SLURM_JOB_ID}.log" 2>&1
}

run_pair "ta ties" &
P1=$!
run_pair "dareties bc" &
P2=$!

wait "$P1"; R1=$?
wait "$P2"; R2=$?
echo "[paired] ta+ties exit=$R1, dareties+bc exit=$R2"
exit $(( R1 || R2 ))
