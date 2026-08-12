#!/bin/bash
#SBATCH --job-name=apr-rpl-glue8
#SBATCH --partition=mcs.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=logs/replay_glue8-%j.out
#SBATCH --error=logs/replay_glue8-%j.out

# Matched-data-budget table on GLUE-8: every method sees the SAME 64 labeled examples
# per task (APR's replay buffer, from the train split). Tier 1 (lambda-search,
# LM-Cocktail, Fisher, head-only) + Tier 2 (Localize-and-Stitch) + the encoder-refinement
# families (ordinary GD, ungated, APR).
#
# Hyperparameters are selected on the REPLAY objective, not the eval split; the script
# also reports the eval-selected (oracle) winner so the selection gap is visible.
# APR lr grid {2,4,8,16}: lr32 is knife-edge on GLUE-8 (see the reproducibility note).
source slurm/common.sh
python scripts/replay_baselines.py --config configs/glue8.yaml \
  --n_probe 64 --steps 5 \
  --lam_grid 0.05 0.1 0.2 0.3 0.4 0.5 \
  --ls_sparsities 0.01 0.05 --ls_gamma 1e-4 --ls_steps 200 --ls_lr 0.1 --ls_bs 16 \
  --head_lr 1e-3 --head_epochs 20 \
  --apr_lrs 2 4 8 16 --nogate_lrs 1 2 4 8 --gd_lrs 5e-5 1e-4 5e-4 1e-3 \
  --out results/compare/replay_baselines_glue8.json "$@"
