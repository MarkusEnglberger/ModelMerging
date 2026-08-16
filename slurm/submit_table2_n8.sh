#!/bin/bash
# Submit the complete n=8+8 counterpart of Table 2: four benchmarks, with a
# from-pretrained job and an all-merge-initializations job for each benchmark.
# Results are written to results/compare/grid_nn_<bench>_<mode>_n8.json.
#
# Usage (from the repository root):
#   bash slurm/submit_table2_n8.sh

set -euo pipefail
: "${PARTITION:=mcs.gpu.q}"

for bench in glue8 clip8 t5_glue8 clip20; do
  for mode in base merges; do
    echo "[submit] Table 2 n=8+8: BENCH=$bench MODE=$mode PARTITION=$PARTITION"
    sbatch --partition="$PARTITION" \
      --export="ALL,BENCH=$bench,MODE=$mode,N=8" slurm/grid_nn.sbatch
  done
done

echo "[submit] submitted all eight jobs"
echo "[submit] after completion: python3 scripts/make_table_n16.py --n 8"
