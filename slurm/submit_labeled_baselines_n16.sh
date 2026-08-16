#!/bin/bash
# Submit the two standalone labeled baselines for all four n=16+16 benchmarks.

set -euo pipefail
: "${PARTITION:=mcs.gpu.q}"

for bench in glue8 clip8 t5_glue8 clip20; do
  echo "[submit] labeled n=16+16 baselines: BENCH=$bench PARTITION=$PARTITION"
  sbatch --partition="$PARTITION" --export="ALL,BENCH=$bench" \
    slurm/labeled_baselines_n16.sbatch
done
