#!/bin/bash
#SBATCH --job-name=ho-cons
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ho_cons-%j.out
#SBATCH --error=logs/ho_cons-%j.out

# RETENTION CONSOLIDATION. Two modes via --export:
#   MODE=nogate            seed 0, ungated lr sweep from base (0.5,1,4,8; lr2
#                          exists) -> turns the single-point "gate helps
#                          retention" observation into a frontier comparison.
#   MODE=seed, SEED=1|2|3  replicate the Pareto-frontier arms on a fresh
#                          buffer draw: APR<-ada, APR<-base (lr4,8),
#                          GD<-base (5e-5,1e-4), ungated<-base lr2, BC merge.
#                          RegMean intentionally omitted (side story).
# Same 14/6 split and S=30 cosine protocol as jobs 6242790/6243949.
set -u
M=${MODE:?set MODE=nogate or MODE=seed}
source slurm/common.sh
if [ "$M" = "nogate" ]; then
  python scripts/heldout_retention.py --config configs/clip20.yaml \
    --heldout sun397 stl10 cifar10 pets food101 flowers102 \
    --n_probe 64 \
    --ta_lams 0.08 --ties_densities --dt_drops --bc_densities \
    --apr_base_lrs --apr_ta_lrs \
    --nogate_base_lrs 0.5 1 4 8 \
    --steps 30 --apr_schedule cosine --apr_order random \
    --out results/compare/heldout_nogate_sweep.json
else
  S=${SEED:?set SEED via --export=ALL,MODE=seed,SEED=...}
  python scripts/heldout_retention.py --config configs/clip20.yaml \
    --heldout sun397 stl10 cifar10 pets food101 flowers102 \
    --n_probe 64 --probe_seed "${S}" \
    --ta_lams 0.08 --ties_densities --dt_drops \
    --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
    --ada --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 \
    --apr_base_lrs 4 8 --apr_ta_lrs --apr_ada_lrs 4 \
    --gd_base_lrs 5e-5 1e-4 --nogate_base_lrs 2 \
    --steps 30 --apr_schedule cosine --apr_order random \
    --out "results/compare/heldout_retention_seed${S}.json"
fi
