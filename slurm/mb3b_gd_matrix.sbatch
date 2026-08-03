#!/bin/bash
#SBATCH --job-name=mb3b-gdmx
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mb3b_gdmx-%j.out
#SBATCH --error=logs/mb3b_gdmx-%j.out

# The decisive regime-map test at 3B (~1.2e7 params per replay example):
# gated APR vs ungated anchored vs ordinary GD, all from the SAME tuned
# TA@l0.5 init (pinned: the selected winner of the composition job's sweep,
# same config -> exact cross-job comparability; apr@lr64 repeated as the
# cross-job anchor cell).
#
# Registered predictions (before results): GD's best cell below both anchored
# arms, worst-task gap larger; at least one GD cell in {3e-4, 1e-3}
# catastrophically below the init; ungated >= gated on mean.
#
# PRIOR EVIDENCE AGAINST prediction 1, recorded for honesty: jobs 24650025 and
# 24739458 (logs only, never written to JSON) both had GD >= APR. They are not
# decisive -- 3 tasks, untuned lambda=0.3, an eval protocol still being repaired
# (the two runs disagree on eval0 for the SAME merge), and APR capped at lr16
# when its optimum later proved to be lr64 -- but they are the reason GD's grid
# below reaches down to 3e-6: GD peaked at 1e-5/5e-5 there, so a grid starting
# at 3e-5 risked edge-clipping the arm this experiment exists to test fairly.
#
# GD gets the WIDE lr grid the CLI help demands. S is the config's 5 for all
# arms; the S=20 "GD just needs more sweeps" follow-up runs only at the 2 best
# GD lrs in a later job (GD runs are not checkpointed, so S=20 costs 4x).

set -euo pipefail
source slurm/common.sh

python scripts/merge_baselines.py \
  --config configs/mergebench_llama3b.yaml \
  --ta_lams 0.5 \
  --skip_families ties dare dareties della bc ls ada \
  --refine_from ta \
  --apr_lrs 64 128 256 \
  --nogate_lrs 16 64 128 256 \
  --control_gd_lrs 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 \
  --out results/compare/mb3b_gd_matrix.json
