#!/bin/bash
#SBATCH --job-name=ho-8tr-n8m
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=logs/heldout_clip20_8train_n8m-%j.out
#SBATCH --error=logs/heldout_clip20_8train_n8m-%j.out
#
# HELD-OUT RETENTION, LITERATURE-BOUNDARY SPLIT AT n=8: train on exactly the
# CLIP-8 suite, hold out the 12 remaining tasks of the 20-task suite.
#
# Why this split. The primary 14/6 study picks its held-out six by ranking base
# zero-shot -- a defensible rule, but an experimenter-chosen one. Here the
# train/held-out boundary is CLIP-8 membership, fixed by the literature, so the
# split involves no choice of ours at all. Of the 12 held-out tasks, seven have
# real headroom above chance (cifar10 .88, stl10 .97, pets .83, food101 .83,
# flowers102 .66, cifar100 .62/chance .01, fashion_mnist .66/chance .1) and are
# the reportable retention probes; the other five are at or near chance for the
# base model (kmnist .10, emnist .11; pcam .556 and rendered_sst2 .586 are
# BINARY, so barely above the .5 coin flip; fer2013 .376 vs .143 is marginal)
# and are recorded but reported separately -- there is nothing to retain on a
# chance-level task. The headroom rule is stated here, before results.
#
# REUSE. Every refinement cell is pinned at the split-selected (lr, S) of the
# corresponding CLIP-8 n=8+8 experiment (grid_nn_clip8_{base,merges}_n8.json):
#   from base:  APR lr16,S50   ungated lr1,S20   GD lr1e-4,S20
#   from TA:    APR lr2,S50
#   from Ada:   APR lr2,S50
# and the merge hyperparameters are those experiments' best_per_family
# (TA l0.3; TIES d0.1,l1; DARE-TIES dd0.5,t0.1,l0.8; BC d0.2,o0.01,l0.4).
# Grid runs persist no weights, so "reuse" = deterministic re-derivation: same
# probe buffers (n=8, probe_seed 0 draws identical indices), same refine() code,
# constant schedule + fixed task order = the grid-run regime. Nothing is tuned
# or selected inside this study; the selections were made by the CLIP-8
# experiments' own selection buffers, so the protocol stays split-selected.
#
# ADAMERGING FIX (v2). The first version of this job used the script's then-
# hardcoded data_key="eval_ds": AdaMerging fitted its coefficients on the FULL
# evaluation sets of the 8 train tasks (87,433 unlabeled images) while every
# other arm saw 8 labeled examples per task. That is the transductive protocol,
# not the matched budget the grid runs and the paper use, so the cell was a
# different model from the merge:ADA-layer-matched that the CLIP-8 n=8 grid
# selected -- breaking the reuse this study depends on. Measured cost of the
# mismatch on the 8 train tasks: transductive +0.569 mean NormRet vs matched
# +0.463, i.e. the protocol was worth +0.106 to the baseline. Every other
# method reproduced the grid exactly (TA +0.270, TIES +0.525 in both), which is
# what isolated AdaMerging as the culprit. Now pinned to probe_buffer.
source slurm/common.sh
python scripts/heldout_retention.py --config configs/clip20.yaml \
  --heldout cifar100 stl10 flowers102 pets pcam fer2013 emnist cifar10 food101 fashion_mnist kmnist rendered_sst2 \
  --n_probe 8 --probe_seed 0 \
  --ta_lams 0.3 \
  --ties_densities 0.1 --ties_lams 1 \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.8 \
  --bc_densities 0.2 --bc_outliers 0.01 --bc_lams 0.4 \
  --ada --ada_data probe_buffer \
  --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 \
  --regmean --regmean_nd 1.0 \
  --apr_base_lrs 16 --apr_base_steps 50 \
  --nogate_base_lrs 1 --nogate_base_steps 20 \
  --gd_base_lrs 0.0001 --gd_base_steps 20 \
  --apr_ta_lrs 2 --apr_ta_steps 50 \
  --apr_ada_lrs 2 --apr_ada_steps 50 \
  --steps 50 --apr_schedule constant \
  --out results/compare/heldout_clip20_8train_n8_adamatched.json
