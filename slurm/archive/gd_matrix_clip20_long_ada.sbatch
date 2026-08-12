#!/bin/bash
#SBATCH --job-name=gdmx20-lada
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gdmx20_long_ada-%j.out
#SBATCH --error=logs/gdmx20_long_ada-%j.out

# LONG-HORIZON + ANNEALED ordinary GD from the AdaMerging-layer init -- the
# STRONGEST non-data-fitted init at 20 tasks (.6651 with init_lam 0.05, job
# 6195612) and the one APR gained most from (.7185). If long+annealed GD is ever
# going to close the gap, a strong init is where it has the best chance, so this
# is the most informative single cell of the long-horizon sweep.
# See gd_matrix_clip20_long_a for the full rationale.
source slurm/common.sh
python scripts/merge_baselines.py --config configs/clip20.yaml \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families ta ties dare dareties bc \
  --refine_from ada \
  --apr_lrs \
  --gd_lrs 5e-5 1e-4 5e-4 1e-3 \
  --gd_steps 30 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 5 --select_by mean_acc \
  --out results/compare/gd_matrix_clip20_long_ada.json "$@"
