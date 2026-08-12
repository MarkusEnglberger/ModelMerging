#!/bin/bash
#SBATCH --job-name=budget-clip20
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=logs/budget_clip20-%j.out
#SBATCH --error=logs/budget_clip20-%j.out

# REPLAY-BUDGET SWEEP on the 20-task CLIP suite: APR and ordinary GD at
# n in {8,16,32} labeled examples per task, with the number of sweeps S TUNED
# rather than fixed. Submit one job per (n, S) cell:
#   sbatch --export=ALL,NPROBE=8,SWEEPS=5   slurm/budget_clip20.sbatch
#   sbatch --export=ALL,NPROBE=8,SWEEPS=30  slurm/budget_clip20.sbatch   (etc.)
# The reported budget curve then takes, per (n, method, init), the best cell
# over BOTH the learning rate and S -- the same best-over-budget rule used at
# n=64, so no method is penalised by an arbitrary horizon.
#
# Why S must not be fixed: at twenty tasks APR is NOT converged at S=5 (S=30
# adds ~+0.01 mean and a large worst-task gain) and GD needs roughly 6x the
# sweeps to reach its own peak. Fixing S=5 would understate both, and the
# optimal S plausibly interacts with n -- fewer examples per sweep may want
# more passes, or may overfit sooner. That interaction is the point.
#
# Why n=64 is absent: it already exists. The one pre-existing small-n cell
# (n=32 from RegMean) is confounded -- it varied the budget, the selection
# protocol and the Gram sample at once -- so it is superseded here.
#
# Grids. S=5 gets WIDE grids to locate the optimum (APR 0.5-8, GD 1e-5..5e-4,
# constant schedule). S=30 is expensive per cell, so it uses grids narrowed
# around the located optimum and the cosine/random-order schedule, which at
# n=64 was best-or-tied for APR and repaired constant-rate GD's unreliability.
# Both grids extend DOWNWARD because the optimal step falls as the buffer
# shrinks (8-task APR: 8 -> 6 -> ~4 over n=64/32/16) and GD's 1e-4 arm already
# collapsed at n=32 from RegMean.
#
# Two invocations: the checkpoint/learned merges, then the base model
# (--ta_lams 0 makes the "merge" theta_0 itself).
set -u
N=${NPROBE:?set NPROBE (8|16|32) via --export=ALL,NPROBE=...}
S=${SWEEPS:?set SWEEPS (5|30) via --export=ALL,SWEEPS=...}
source slurm/common.sh
echo "[budget] n_probe=${N} steps=${S}"

if [ "${S}" -le 5 ]; then
  APR_LRS="0.5 1 2 4 8"; GD_LRS="1e-5 5e-5 1e-4 5e-4"; SCHED="constant"
else
  APR_LRS="0.5 1 2 4";   GD_LRS="1e-5 5e-5 1e-4";       SCHED="cosine"
fi
echo "[budget] apr_lrs=${APR_LRS} gd_lrs=${GD_LRS} schedule=${SCHED}"

python scripts/merge_baselines.py --config configs/clip20.yaml \
  --n_probe "${N}" \
  --ta_lams 0.08 \
  --ties_densities 0.1 --ties_lams 0.6 \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families dare dareties bc \
  --refine_from ta ties ada \
  --apr_lrs ${APR_LRS} --apr_schedules ${SCHED} --apr_order random \
  --gd_lrs ${GD_LRS} --gd_steps "${S}" --gd_schedules ${SCHED} \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps "${S}" --select_by mean_acc \
  --out "results/compare/budget_clip20_n${N}_S${S}.json"

python scripts/merge_baselines.py --config configs/clip20.yaml \
  --n_probe "${N}" \
  --ta_lams 0 \
  --skip_families ties dare dareties bc ada \
  --refine_from ta \
  --apr_lrs ${APR_LRS} --apr_schedules ${SCHED} --apr_order random \
  --gd_lrs ${GD_LRS} --gd_steps "${S}" --gd_schedules ${SCHED} \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps "${S}" --select_by mean_acc \
  --out "results/compare/budget_clip20_n${N}_S${S}_base.json"
