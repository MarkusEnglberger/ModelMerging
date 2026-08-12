#!/bin/bash
#SBATCH --job-name=ms20-tiesada
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ms20_tiesada-%j.out
#SBATCH --error=logs/ms20_tiesada-%j.out

# MULTI-SEED replication, TIES + ADA inits. Submit once per replay seed:
#   sbatch --export=ALL,SEED=1 slurm/ms_clip20_tiesada.sbatch   (seeds 1..4;
#   seed 0 = the existing single-seed campaign).
# Protocol: vary ONLY the replay-buffer seed (--probe_seed); every
# hyperparameter is PINNED at its seed-0-selected value, so the numbers are
# the seed-variance of the reported cells, uncontaminated by re-selection
# (the holdout pilot showed the selection gap itself is ~0). cfg.seed and the
# random task order stay at 0.
# Pinned cells: apr S30 lr4 cosine (headline), apr S5 lr4 const
# (matched-budget), nogate S5 lr1 (gate-neutrality), gd S30 1e-4
# constant+cosine (scoreboard reference).
set -u
S=${SEED:?set SEED via --export=ALL,SEED=...}
source slurm/common.sh
echo "[ms] probe_seed=${S} ties+ada"

python scripts/merge_baselines.py --config configs/clip20.yaml \
  --probe_seed "${S}" \
  --ties_densities 0.1 --ties_lams 0.6 \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families ta dare dareties bc ls \
  --refine_from ties ada \
  --apr_lrs 4 --apr_schedules cosine --apr_order random \
  --gd_lrs 1e-4 --gd_steps 30 --gd_schedules constant cosine \
  --gd_lr_min_frac 0.05 --gd_order random \
  --steps 30 --select_by mean_acc \
  --out "results/compare/ms_clip20_tiesada_S30_seed${S}.json"

python scripts/merge_baselines.py --config configs/clip20.yaml \
  --probe_seed "${S}" \
  --ties_densities 0.1 --ties_lams 0.6 \
  --ada_variants layer --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 \
  --ada_bs 16 --ada_workers 0 \
  --skip_families ta dare dareties bc ls \
  --refine_from ties ada \
  --apr_lrs 4 --apr_schedules constant \
  --nogate_lrs 1 \
  --steps 5 --select_by mean_acc \
  --out "results/compare/ms_clip20_tiesada_S5_seed${S}.json"
