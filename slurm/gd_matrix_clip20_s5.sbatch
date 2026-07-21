#!/bin/bash
#SBATCH --job-name=gdmx20-s5
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/gdmx20_s5-%j.out
#SBATCH --error=logs/gdmx20_s5-%j.out

# INIT x METHOD MATRIX, matched budget (S=5, constant lr) -- the decisive test of
# "does APR beat ordinary GD consistently across initializations?"
#
# Motivation: an audit of all 20 result files found EVERY GD cell in the project
# is from the TA init. GD has never been run from TIES/ADA/DARETIES/BC/RegMean,
# so the cross-init claim has no evidence. This runs apr / nogate / GD from the
# SAME five inits in ONE report, with identical budget and setup.
#
# Each family is PINNED to its best-known 20-task config (no re-sweeping; compute
# goes into the GD grid instead):
#   TA l0.08 (.6043) | TIES d0.1,l0.6 (.6262) | DARETIES dd0.5,t0.1,l0.4 (.6168)
#   BC d0.1,o0.01,l0.2 (.6110) | ADA-layer @ init_lam 0.05 (.6651)
# GD gets a DELIBERATELY WIDE lr grid (1e-5..5e-3, 3 decades): its optimum was
# only ever tuned at TA (1e-4), and from a farther init the optimum may move, so
# a narrow grid would unfairly favour APR. Grid-edge warnings apply as usual.
# RegMean init is covered separately by the regmean_clip20_controls job.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0.08 \
  --ties_densities 0.1 --ties_lams 0.6 \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.4 \
  --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families dare \
  --refine_from ta ties dareties bc ada \
  --apr_lrs 2 4 8 \
  --nogate_lrs 0.5 1 2 \
  --gd_lrs 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 \
  --gd_steps 5 --gd_schedules constant \
  --steps 5 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_s5.json "$@"
