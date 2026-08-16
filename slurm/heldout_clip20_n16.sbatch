#!/bin/bash
#SBATCH --job-name=ho-c20-n16
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --output=logs/heldout_clip20_n16-%j.out
#SBATCH --error=logs/heldout_clip20_n16-%j.out
#
# HELD-OUT RETENTION AT n=16 -- re-run of the retired n=64 study under the
# paper's current small-budget protocol.
#
# Why re-run. The n=64 held-out study (retired from the paper along with the
# rest of the evaluation-selected n=64 tier) was the only DIRECT measurement of
# whether refinement preserves pretrained capability: merge/refine on 14
# specialized tasks, evaluate the 6 strongest natural-image tasks purely
# zero-shot through frozen text-derived heads, so the held-out axis is exactly
# "how much of theta_0's own ability survived". It produced the paper's
# accuracy--retention Pareto frontier, and it was the one setting where the
# GATE was clearly beneficial rather than neutral. Everything in the paper is
# now n<=16, so this rebuilds it at the surviving budget.
#
# Protocol note. This is a frontier/arc measurement, not a selected cell: each
# arm is swept over its learning rate and the whole arc is reported, so no
# configuration is picked using the evaluation split. That keeps it compatible
# with the 16+16 split-selection protocol without needing a selection buffer --
# there is nothing being selected. State this explicitly in the caption.
#
# Arms carried over from the archived full-coverage run (heldout_clip20_full),
# with lrs re-centred for the smaller buffer: the 20-task budget study found the
# optimal APR rate falls to eta ~1-4 as n shrinks, so the base sweep is widened
# downward (2 4 8) rather than pinned at the n=64 optimum. GD keeps both of its
# rates -- it travels far less than APR (.09-.22 vs .44-.59) and may retain
# BETTER while scoring worse, which is the caveat the paper must own. nogate is
# the arm that decides whether the gate's retention benefit replicates at n=16.
#
# S=30 cosine/random is kept from the n=64 run so the two are comparable; the
# 20-task budget study showed small-n cells want S>=30 anyway.
source slurm/common.sh
python scripts/heldout_retention.py --config configs/clip20.yaml \
  --heldout sun397 stl10 cifar10 pets food101 flowers102 \
  --n_probe 16 --probe_seed 0 \
  --ta_lams 0.08 0.1 0.15 0.2 \
  --ties_densities 0.1 --ties_lams 0.6 0.8 \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.4 \
  --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
  --ada --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 \
  --regmean --regmean_nd 1.0 \
  --apr_base_lrs 2 4 8 --apr_ta_lrs 4 --apr_ada_lrs 4 \
  --gd_base_lrs 5e-5 1e-4 --nogate_base_lrs 2 \
  --steps 30 --apr_schedule cosine --apr_order random \
  --out results/compare/heldout_retention_clip20_n16.json
