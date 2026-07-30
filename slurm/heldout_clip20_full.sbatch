#!/bin/bash
#SBATCH --job-name=ho-c20-full
#SBATCH --partition=mcs.gpu.q,tue.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --exclude=tue-gpua001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=logs/heldout_clip20_full-%j.out
#SBATCH --error=logs/heldout_clip20_full-%j.out

# HELD-OUT RETENTION, FULL BASELINE COVERAGE (extends job 6242790, which had
# only base / TA / TIES / APR). Same 14-train / 6-held-out split; every merge
# and every refinement step sees ONLY the 14 train tasks; the 6 held-out tasks
# are evaluated zero-shot through their frozen text-derived heads.
#
# Added arms and why:
#   AdaMerging-14  strongest data-free merge in the 20-task tables (.665 vs TA
#                  .604); also the sharpest test of "tune only on what you
#                  merged" -- it LEARNS per-layer lambda by entropy-min on the
#                  train tasks, exactly the recipe whose hand-tuned analogue
#                  (the TA lambda sweep) cost 21 points of held-out accuracy.
#   RegMean-14     independent test of our own null-space account: if it sits
#                  ~450 from base yet retains fine, that displacement really is
#                  functionally inert.
#   DARE-TIES, BC  complete the checkpoint-only tier of the paper's taxonomy.
#   GD-from-base   the paper's own from-base comparison on the retention axis.
#                  GD travels much less than APR (.09-.22 vs .44-.59), so it may
#                  retain BETTER while scoring worse -- a caveat worth owning.
#   nogate-base    does the gate ~ anchor equivalence hold for retention too?
#   APR<-ada14     does refining from a STRONG merge cost retention the way it
#                  did from TA (-.008 held for +.118 train)?
# lrs pinned at the 20-task optima for each family. S=30 cosine/random.
source slurm/common.sh
python scripts/heldout_retention.py --config configs/clip20.yaml \
  --heldout sun397 stl10 cifar10 pets food101 flowers102 \
  --n_probe 64 \
  --ta_lams 0.08 0.1 0.15 0.2 \
  --ties_densities 0.1 --ties_lams 0.6 0.8 \
  --dt_drops 0.5 --dt_trims 0.1 --dt_lams 0.4 \
  --bc_densities 0.1 --bc_outliers 0.01 --bc_lams 0.2 \
  --ada --ada_init_lam 0.05 --ada_steps 300 --ada_lr 1e-3 --ada_bs 16 \
  --regmean --regmean_nd 1.0 \
  --apr_base_lrs 4 8 --apr_ta_lrs 4 --apr_ada_lrs 4 \
  --gd_base_lrs 5e-5 1e-4 --nogate_base_lrs 2 \
  --steps 30 --apr_schedule cosine --apr_order random \
  --out results/compare/heldout_retention_clip20_full.json
