#!/bin/bash
#SBATCH --job-name=gdmx20-apr30
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/gdmx20_apr30-%j.out
#SBATCH --error=logs/gdmx20_apr30-%j.out

# FAIRNESS CELLS: APR at S=30 from the TIES and ADA inits, constant AND
# cosine-annealed (random task order).
# Motivation 1 (budget): gd_matrix_clip20_long_a showed GD-from-TIES @S30
# (.7057/.3841) BEATS apr-from-ties @S5 (.6973) -- a 6x budget mismatch. If
# APR@S30 also improves, the asymptote claim survives; if APR is flat S5->S30
# (clip8: +.017), the honest claim is efficiency.
# Motivation 2 (symmetry, user's point): the long GD arms got cosine+random
# order but every 20-task APR cell so far is fixed-order/constant-lr -- yet
# annealing+random order is the natural mitigation for APR's sequential
# thrash (kmnist step norms stay flat ~0.016/sweep at S=5). These arms remove
# that asymmetry: --apr_schedules constant cosine, cosine arms use random
# order (mirroring the GD arms exactly).
# Also first run recording replay_obj per refine cell -> replay-honest
# hyperparameter selection becomes possible for this report.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ties_densities 0.1 --ties_lams 0.6 \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families ta dare dareties bc \
  --refine_from ties ada \
  --apr_lrs 2 4 \
  --apr_schedules constant cosine --apr_order random \
  --gd_lr_min_frac 0.05 \
  --steps 30 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_apr30.json "$@"
