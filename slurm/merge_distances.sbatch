#!/bin/bash
#SBATCH --job-name=merge-dist
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_distances-%j.out
#SBATCH --error=logs/merge_distances-%j.out
#
# Measure ||merge - theta_0|| for each family's split-selected merge, on the
# three encoder benchmarks. Needed for the absolute-frame displacement figure:
# the grid runs record merge displacement relative to the config-lambda merge,
# not relative to theta_0, and only task arithmetic (linear in lambda) can be
# converted analytically. Everything else has to be measured, which is why the
# figure previously carried a TA reference point and nothing else.
#
# No evaluation is performed by the script itself -- it is weight arithmetic --
# but MergeContext.build scores base/expert once unless the eval0 cache is warm,
# which is the only reason this asks for a GPU and four hours.
#
# AdaMerging is skipped: it would have to be re-fit, not rebuilt.
source slurm/common.sh

python scripts/merge_distances.py --config configs/glue8.yaml \
  --grid results/compare/grid_nn_glue8_merges_n16.json \
  --n_probe 16 --probe_seed 0 \
  --out results/compare/merge_distances_glue8.json

python scripts/merge_distances.py --config configs/clip8.yaml \
  --grid results/compare/grid_nn_clip8_merges_n16_s200_tatiesdareties.json \
  --n_probe 16 --probe_seed 0 \
  --out results/compare/merge_distances_clip8.json

python scripts/merge_distances.py --config configs/clip20.yaml \
  --grid results/compare/grid_nn_clip20_merges_n16_fast.json \
  --n_probe 16 --probe_seed 0 \
  --out results/compare/merge_distances_clip20.json

echo "[done] all three merge-distance measurements"
