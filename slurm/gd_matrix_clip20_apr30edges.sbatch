#!/bin/bash
#SBATCH --job-name=gdmx20-a30edge
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/gdmx20_apr30edges-%j.out
#SBATCH --error=logs/gdmx20_apr30edges-%j.out

# BRACKET THE S=30 APR EDGES. The apr30 job's {2,4} grid left 5 of 6 curves at
# edges (ties/ada: constant peaks at 2 = low edge, cosine at 4 = high edge;
# regmean constant peaks at 0.25 = low edge). All current S30 APR numbers are
# therefore lower bounds -- biased AGAINST APR in the any-budget parity table.
# Extends each init's grid one step past its edge in both directions.
source slurm/common.sh

# ties + ada: lr {1, 8} x {constant, cosine} completes {1,2,4,8} coverage
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ties_densities 0.1 --ties_lams 0.6 \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families ta dare dareties bc ls \
  --refine_from ties ada \
  --apr_lrs 1 8 \
  --apr_schedules constant cosine --apr_order random \
  --gd_lr_min_frac 0.05 \
  --steps 30 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_apr30edges.json

# from-base: bracket the S30 cosine high edge (curve still rising at lr8:
# 2:.6554 4:.6710 8:.6831); from theta_0 the |v_i| are full task vectors, so
# the optimum plausibly sits at clip8-like lrs (12-16+)
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ta_lams 0 \
  --skip_families ties dare dareties bc ls ada \
  --refine_from ta \
  --apr_lrs 12 16 \
  --apr_schedules cosine --apr_order random \
  --gd_lr_min_frac 0.05 \
  --steps 30 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_base_s30_hi.json

# regmean: bracket the constant low edge (0.125); cosine already interior at 0.5
python scripts/regmean_baseline.py --config configs/clip20.yaml \
  --n_probe 64 --gram_ns 64 --nondiag_scales 1.0 \
  --apr_lrs 0.125 \
  --apr_schedules constant \
  --gd_lr_min_frac 0.05 \
  --steps 30 --select_by mean_acc \
  --out results/compare/regmean_clip20_aprlong_edge.json
